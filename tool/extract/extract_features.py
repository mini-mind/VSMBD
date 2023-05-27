import os
from importlib import import_module

import einops
import numpy as np
import torch
from tqdm import tqdm

from .config import cfg, full_vids

Model = getattr(import_module("tool.extract.model."+cfg.model), 'Model')
parameters = None
if cfg.get("load_path"):
    parameters = torch.load(cfg.load_path, 'cpu')
    if cfg.get("load_key"):
        parameters = parameters[cfg.load_key]
model = Model(parameters)
model.requires_grad_(False)
model.eval().to(cfg.DEVICE)

Dataset = getattr(import_module("tool.extract.dataset."+cfg.dataset), 'Dataset')
dataloader = torch.utils.data.DataLoader(
    dataset=Dataset(full_vids),
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    drop_last=False,
    pin_memory=(cfg.get('DEVICE', 'cpu')!='cpu')
)

print("Saving path:", cfg.save_path)
if cfg.save_path.endswith('.h5'):
    import h5py
    with h5py.File(cfg.save_path, 'w') as f:
        for vid in full_vids:
            grp = f.create_group(vid)
        for batch in tqdm(dataloader):
            data = einops.rearrange(batch['data'], 'b k c h w -> (b k) c h w').to(cfg.DEVICE)
            batch_feats = einops.reduce(model(data).detach().cpu(), '(b k) d -> b d', 'max', k=3)
            for vid, shid, feats in zip(batch['vid'], batch['shid'], batch_feats):
                grp = f[vid]
                grp.create_dataset(f'{shid.item():04d}', data=feats)
else:
    raise NotImplementedError("Can not save as this type")