import torch.nn as nn
import torch
import math
from einops import rearrange
from models.modules.dit import DiTBlock, RotaryEmbedding
 
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
 
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
 
    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale[:, None]) + shift[:, None]
        x = self.linear(x)
        return x
    
class DiT(nn.Module):
    def __init__(
        self, 
        hidden_dim=768, 
        spk_dim=256,
        num_layers=8, 
        num_heads=12, 
        ff_mult=3,
        n_quantizers=4
    ):
        super().__init__()
        
        self.n_quantizers = n_quantizers
        self.time_embed = TimestepEmbedding(hidden_dim)

        self.spk_embed = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.segment_embed = nn.Embedding(3, hidden_dim)
        self.register_buffer("segment_ids", torch.tensor([0, 1, 1, 2, 2, 2]))
        
        self.x_proj = nn.Linear((n_quantizers + 2) * hidden_dim, hidden_dim) # + 2 for content
 
        # DiT Blocks
        cond_dim = hidden_dim
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, cond_dim, num_heads, ff_mult)
            for _ in range(num_layers)
        ])
 
        # Long skip connection
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
 
        # Rotary embeddings
        self.rotary = RotaryEmbedding(hidden_dim // num_heads)

        # Final layer
        self.final_layer = FinalLayer(hidden_dim, n_quantizers * hidden_dim)

    def forward(
        self, 
        x, 
        content_emb, 
        prompt_emb,
        spk_emb, 
        t, 
        audio_masks=None
    ):
        """
        Args:
            x: [B, 4, L, D]
            content_emb: [B, 2, L, D]
            prompt_emb: [B, 6, L_p, D]
            spk_emb: [B, spk_dim]
            t: [B,]
        """
        split_len = prompt_emb.shape[2]
        t = self.time_embed(t) # [B, hidden_dim]
        spk = self.spk_embed(spk_emb) # [B, hidden_dim]

        cond = t + spk

        x = torch.cat([x[:, :1], content_emb, x[:, 1:]], dim=1)
        x = torch.cat([prompt_emb, x], dim=2)
        segment_bias = self.segment_embed(self.segment_ids).unsqueeze(0).unsqueeze(2)
        x = x + segment_bias
        x = rearrange(x, 'b n l d -> b l (n d)', n=self.n_quantizers + 2)
        x = self.x_proj(x)

        if audio_masks is not None:
            audio_masks = torch.cat([
                torch.zeros((audio_masks.shape[0], split_len), dtype=audio_masks.dtype, device=audio_masks.device),
                audio_masks
            ], dim=1)

        # Rotary positions
        seq_len = x.shape[1]
        rotary_pos = self.rotary(seq_len, x.device)
 
        # Long skip connection
        x_skip = self.proj_in(x)

        # Process through DiT blocks
        for block in self.blocks:
            x = block(x, cond, rotary_pos, audio_masks)
 
        # Final projection and skip
        x = x_skip + x
        x = self.final_layer(x, cond)  # [B, seq_len, 4D]
        x = x[:, split_len:, :]
 
        return x