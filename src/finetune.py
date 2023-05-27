from .common.config import *
load_config('finetune')
import torch, os, numpy as np, time
from tqdm import tqdm
from .common.metrics import Metrics
from .common.logger import log
from .runtime import *
from omegaconf import OmegaConf as OC


metrics = Metrics(data.vno_shot_bound)

for epoch in range(OC.select(cfg, 'finetune.params.from_epoch'), OC.select(cfg, 'finetune.params.total_epochs')):
    train_loss_log, val_loss_log = [], []
    
    model.train()
    progress = tqdm(train_loader, f'{epoch=}')
    for i, payload in enumerate(progress, 1):
        optimizer.zero_grad()
        out = model.finetune(payload)
        loss = out['loss']
        loss.backward()
        train_loss_log.append(loss.item())
        optimizer.step()
        lr = lr_scheduler.get_last_lr()[0]
        progress.set_postfix(lr=f'{lr:.8f}', loss=f'{loss.item():.8f}')
        lr_scheduler.step()
        
    model.eval()
    loss_cache = []
    with torch.no_grad():
        for i,payload in enumerate(tqdm(val_loader, 'validating')):
            out = model.predict(payload)
            loss = out['loss']
            logits = out['logits']
            val_loss_log.append(loss.item())
            # update metric
            vnos = payload["vno"]
            sids = payload["center_shid"]
            labels = payload['center_label']
            metrics.update(vnos, sids, logits, labels)
            
    score = metrics.compute()
    score.update({
        'train_loss': np.mean(train_loss_log),
        'val_loss': np.mean(val_loss_log),
    })
    metrics.reset()
    log()
    log(f'{epoch=}')
    for k,v in score.items():
        log(k, ':', v)
        
    if OC.select(cfg, 'base.save_root'):
        torch.save({
            'parameters':model.state_dict(),
            'score':score
        }, os.path.join(cfg.base.checkpoint_dir, f'{epoch=}.pt'))