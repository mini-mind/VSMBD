import torch, einops
from torch import nn
from ..common.config import cfg

class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=128):
        super().__init__()
        self.size = size
        self.pe = nn.Embedding(size, dim)
        self.pos_ids = torch.arange(size, dtype=torch.long, device=cfg.base.DEVICE)
        
    def forward(self, x):
        pos_ids = einops.repeat(self.pos_ids, 'n -> b n', b=len(x))
        embeddings = torch.cat([x, self.pe(pos_ids)], dim=-1)
        return embeddings