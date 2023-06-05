from .common.config import *
load_config('evaluate')
import torch, os, numpy as np, time
from tqdm import tqdm
from .common.metrics import Metrics
from .common.logger import log
from .runtime import *
from omegaconf import OmegaConf as OC
import pandas as pd


metrics = Metrics(data.vno_shot_bound)
# evaluate the performance of the specified model on the specified dataset
val_loss_log = []
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
            'val_loss': np.mean(val_loss_log),
        })
        results = {
            'probs':metrics.probs,
            'gts':metrics.gts,
            'vids':[data.vno2vid[vno.item()] for vno in metrics.vnos],
            'shids':metrics.sids
        }
        metrics.reset()
        log()
        for k,v in score.items():
            log(k, ':', v)
        # file_name = f'{os.path.basename(cfg.evaluate.load_path).split(".")[0]}.pkl'
        # pd.to_pickle(results, os.path.join(cfg.base.save_root, cfg.base.name, file_name))