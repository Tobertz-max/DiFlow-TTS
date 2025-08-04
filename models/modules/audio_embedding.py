import torch
import torch.nn as nn
from einops import rearrange

class AudioEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, padding_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.audio_embed = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx) for _ in range(2)
        ])
        
    def forward(self, codes):
        # codes: [B, 4L]
        codes = rearrange(codes, 'b (n l) -> b n l', n=4) # [B, 4, L]
        chunks = torch.split(codes, [1, 3], dim=1)

        embs_list = []
        for chunk, audio_emb in zip(chunks, self.audio_embed):
            embs = audio_emb(chunk)
            embs_list.append(embs)

        embs = torch.cat(embs_list, dim=1) # [B, 4, L, D]
        embs = rearrange(embs, 'b n l d -> b (n l) d', n=4)
        return embs