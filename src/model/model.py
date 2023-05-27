import torch, vit_pytorch, einops, numpy as np
from torch import nn
from torch.nn import functional as F
from ..common.config import cfg
from .mlp import MlpHead
from .emb_position import PositionEmbedding
from .visual_encoder import VisualEncoder

class ContextEncoder(nn.Module):
    def __init__(self, shot_dim, pe_dim, heads=8, mlp_dim=3072):
        super().__init__()
        self.PE = PositionEmbedding(cfg.base.params.clip_len, pe_dim)
        self.shot_dim = shot_dim

        dim = shot_dim + pe_dim
        self.emb = MlpHead(hid_dim=shot_dim, out_dim=shot_dim, bn=True)
        self.tf = vit_pytorch.vit.Transformer(
            dim=dim,
            depth=2,
            heads=heads,
            dim_head=dim//heads,
            mlp_dim=mlp_dim,
            dropout=cfg.base.params.dropout,
        )

    def forward(self, x):
        '''
        in:
            emb: [b, n=cfg.base.params.clip_len, d=2048]
        out: [b n dim=1024]
        '''
        x = self.emb(x)
        x = self.PE(x) # [b n dim]
        x = self.tf(x)
        return x

class Model(nn.Module):
    def __init__(self, shot_dim=896, pe_dim=128):
        super().__init__()
        # utils
        self.eps = 1e-6
        self.batch_indices = torch.arange(cfg[cfg.base.stage].params.batch_size).to(cfg.base.DEVICE)
        # modules
        if cfg.base.path.get('keyframes_root', False):
            self.scrach = VisualEncoder(False)
        self.encoder = ContextEncoder(shot_dim, pe_dim, heads=8)
        self.pp_head = MlpHead()
        self.sbd_head = MlpHead()
    
    def pretrain(self, batch):
        emb = batch['video'].to(cfg.base.DEVICE)
        if cfg.base.path.get('keyframes_root', False):
            images = batch['images'].to(cfg.base.DEVICE)
            semb = self.scrach(images)
            emb = torch.cat([emb, semb], -1)
        emb = self.encoder(emb) # [b n d=shot_dim+pe_dim]
        pos = batch['pos_idx'].to(cfg.base.DEVICE)
        neg = batch['neg_idx'].to(cfg.base.DEVICE)
        pp_loss = self.pp_loss(emb, pos, neg)
        out = {
            'emb': emb,
            'loss': pp_loss
        }
        return out

    def cat_duration(self, emb, batch):
        emb = emb[:, cfg.base.params.clip_len//2]
        if 'duration' in batch.keys():
            duration = batch['duration'].to(cfg.base.DEVICE)
            emb = torch.cat([emb, duration], dim=-1)
        return emb

    def finetune(self, batch):
        emb = batch['video'].to(cfg.base.DEVICE)
        emb = self.encoder(emb) # [b n d=shot_dim+pe_dim]
        labels = batch['center_label'].to(cfg.base.DEVICE) # [b]
        emb = self.cat_duration(emb, batch)
        logits = self.sbd_head(emb)
        ce_loss = self.ce_loss(logits, labels)
        out = {
            'emb': emb,
            'loss': ce_loss
        }
        return out

    def predict(self, batch):
        emb = batch['video'].to(cfg.base.DEVICE)
        emb = self.encoder(emb) # [b n d=shot_dim+pe_dim]
        labels = batch['center_label'].to(cfg.base.DEVICE) # [b]
        emb = self.cat_duration(emb, batch)
        logits = self.sbd_head(emb)
        ce_loss = self.ce_loss(logits, labels)
        out = {
            'emb': emb,
            'logits': logits,
            'loss': ce_loss
        }
        return out
        
    def predict_pp(self, batch):
        emb = batch['video'].to(cfg.base.DEVICE)
        emb = self.encoder(emb) # [b n d=shot_dim+pe_dim]
        labels = batch['center_label'].to(cfg.base.DEVICE) # [b]
        logits = self.pp_head(emb[:, cfg.base.params.clip_len//2])
        ce_loss = self.ce_loss(logits, labels)
        out = {
            'emb': emb,
            'logits': logits,
            'loss': ce_loss
        }
        return out
    
    def regular(self, logits):
        return torch.mean(logits*logits)

    def pp_loss(self, emb, pos, neg):
        b, n, *_ = emb.shape
        labels = torch.zeros(len(pos)*2, dtype=torch.long, device=cfg.base.DEVICE)
        labels[:b] = 1
        emb = torch.cat([emb[self.batch_indices, pos], emb[self.batch_indices, neg]])
        logits = self.pp_head(emb)
        loss = F.cross_entropy(logits, labels)
        return loss + self.regular(logits)
        
    def ce_loss(self, emb, labels):
        loss = F.cross_entropy(emb, labels, reduction="none")
        lpos = labels == 1
        lneg = labels == 0
        wp = lpos / (lpos.sum() + self.eps)
        wn = lneg / (lneg.sum() + self.eps)
        w = (wp + wn) / 2
        loss = (w * loss).sum()
        return loss