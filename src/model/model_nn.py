import torch, vit_pytorch, einops, numpy as np
from torch import nn
from torch.nn import functional as F
from ..common.config import cfg
from munch import Munch

from .head import MlpHead

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # utils
        self.eps = 1e-6
        self.batch_indices = torch.arange(cfg[cfg.base.stage].params.batch_size).to(cfg.base.DEVICE)
        # modules
        self.sbd_head = MlpHead(clip_len=cfg.base.params.clip_len)
    
    def finetune(self, batch):
        emb = batch['video'].to(cfg.base.DEVICE)
        labels = batch['center_label'].to(cfg.base.DEVICE) # [b]
        logits = self.sbd_head(emb)
        ce_loss = self.ce_loss(logits, labels)
        out = {
            'emb': emb,
            'loss': ce_loss
        }
        return out

    def predict(self, batch):
        emb = batch['video'].to(cfg.base.DEVICE)
        labels = batch['center_label'].to(cfg.base.DEVICE) # [b]
        logits = self.sbd_head(emb)
        ce_loss = self.ce_loss(logits, labels)
        out = {
            'emb': emb,
            'logits': logits,
            'loss': ce_loss
        }
        return out
        
        
    def ce_loss(self, emb, labels):
        loss = F.cross_entropy(emb, labels, reduction="none")
        lpos = labels == 1
        lneg = labels == 0
        wp = lpos / (lpos.sum() + self.eps)
        wn = lneg / (lneg.sum() + self.eps)
        w = (wp + wn) / 2
        loss = (w * loss).sum()
        return loss