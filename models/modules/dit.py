import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
 
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)
    def apply_rope(self, pos, t):
        """
        Args:
            pos: positional encoding of shape [seq_len, head_dim]
            t: tensor of shape [batch, seq_len, num_heads, head_dim]
        Returns:
            Tensor with RoPE applied.
        """
        # Explicitly reshape pos to broadcast over batch and num_heads
        # now shape: [1, seq_len, 1, head_dim]
        pos = pos.unsqueeze(0).unsqueeze(2)
        return t * pos.cos() + self._rotate_half(t) * pos.sin()
    
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, num_heads, ff_mult):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 6)
        )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # head dimension
        # Self-attention components
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.rotary = RotaryEmbedding(self.head_dim)
        # Cross-attention components
        # self.norm2 = nn.LayerNorm(hidden_dim)
        # self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        # Gated MLP
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Linear(ff_mult * hidden_dim, hidden_dim)
        )
    def forward(self, x, cond, rotary_pos, padding_audio=None):
        batch_size, seq_len, _ = x.shape
        # ===== Self-attention with RoPE =====
        residual = x
        shift_msa1, scale_msa1, gate_msa1, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        x = self.norm1(x) * (1 + scale_msa1[:, None]) + shift_msa1[:, None]
        d_model = x.size(-1)  # should be hidden_dim
        w = self.attn.in_proj_weight  # shape: [3*d_model, d_model]
        b = self.attn.in_proj_bias    # shape: [3*d_model]
        q = F.linear(x, w[:d_model, :], b[:d_model])
        k = F.linear(x, w[d_model:2*d_model, :], b[d_model:2*d_model])
        v = F.linear(x, w[2*d_model:, :], b[2*d_model:])
        # Reshape q, k, v to [batch, seq_len, num_heads, head_dim]
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.num_heads)
        # -- Apply RoPE to q and k.
        q = self.rotary.apply_rope(rotary_pos, q)  
        k = self.rotary.apply_rope(rotary_pos, k)
        # Permute to [batch, num_heads, seq_len, head_dim] for attention computation
        q = q.permute(0, 2, 1, 3)  
        k = k.permute(0, 2, 1, 3)  
        v = v.permute(0, 2, 1, 3)
 
        if padding_audio is not None:
            attn_mask = (~padding_audio).unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(x.shape[0], self.num_heads, q.shape[-2], k.shape[-2])
        else:
            attn_mask = None  
        # Scaled Dot-Product Attention --
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=0.0)
        # Merge heads back: reshape to [batch, seq_len, hidden_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, d_model)
        x = gate_msa1[:, None] * attn_out + residual
        # ===== Cross-attention =====
        # residual = x
        # x = self.norm2(x) * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        # cross_attn_out = self.cross_attn(
        #     query=x.transpose(0, 1),
        #     key=content_emb.transpose(0, 1),
        #     value=content_emb.transpose(0, 1),
        #     key_padding_mask=padding_audio if padding_audio is not None else None
        # )[0].transpose(0, 1)
 
        # x = gate_msa2[:, None] * cross_attn_out + residual
        # ===== Gated MLP =====
        residual = x
        x = self.norm3(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        mlp_out = self.mlp(x)
        x = gate_mlp[:, None] * mlp_out + residual
        return x