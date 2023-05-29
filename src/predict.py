from .common.config import *
load_config('evaluate')
import torch, os, numpy as np, time
from tqdm import tqdm
from .common.metrics import Metrics
from .common.logger import log
from .runtime import *
from omegaconf import OmegaConf as OC
import pandas as pd
from torch.nn import functional as F


results = {}

if not os.path.exists(cfg.evaluate.load_path):
    print('No checkpoint:', cfg.evaluate.load_path)
else:
    state_dict = torch.load(cfg.evaluate.load_path, 'cpu')
    model.load_state_dict(state_dict['parameters'])
    model.eval()
    if OC.select(cfg, 'evaluate.head') == 'sbd':
        predict = model.predict
    elif OC.select(cfg, 'evaluate.head') == 'pp':
        predict = model.predict_pp

    with torch.no_grad():
        for i,payload in enumerate(tqdm(val_loader, 'Evaluating')):
            out = predict(payload)
            logits = out['logits']
            # update metric
            vnos = payload["vno"]
            sids = payload["center_shid"]
            # metrics.update(vnos, sids, logits, labels)
            logits = F.softmax(logits.cpu(), dim=1)[:, 1]

            for vno, sid, logit in zip(vnos, sids, logits):
                vid = data.vno2vid[vno.item()]
                sid = sid.item()
                logit = logit.item()
                vid_data = results.get(vid, None)
                if vid_data is None:
                    results[vid] = {}
                sid_data = results[vid].get(sid, None)
                if sid_data is None:
                    results[vid][sid] = logit

            # print(vid, sid, logit)

        pd.to_pickle(results, os.path.join(cfg.base.save_root, cfg.base.name, 'evaluate', cfg.base.name+'.pkl'))