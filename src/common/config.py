import os, sys
import numpy as np
import pandas as pd
import torch
import json
from munch import Munch
from omegaconf import OmegaConf as OC
import argparse
import random


# read config
for yaml in sys.argv:
    if yaml.endswith('.yaml'):
        break
print('Configuration:', yaml)
cfg = OC.load(yaml)
cfg.merge_with_cli()
data = Munch()


def load_config(stage, cfg=cfg):
    # base config
    torch.backends.cudnn.benchmark = True
    cfg.base.stage = stage
    if OC.select(cfg, 'base.save_root'):
        cfg.base.checkpoint_dir = os.path.join(cfg.base.save_root, cfg.base.name, cfg.base.stage)
        os.makedirs(cfg.base.checkpoint_dir, exist_ok=True)

    # load data
    if OC.select(cfg, 'base.path.shot_path'):
        data.shot_bound = pd.read_pickle(cfg.base.path.shot_path)
    if OC.select(cfg, 'base.path.scene_path'):
        data.anno = pd.read_pickle(cfg.base.path.scene_path)
    if OC.select(cfg, 'base.path.label_path'):
        data.label_dict = pd.read_pickle(cfg.base.path.label_path)
        data.scene_dict = {vid:np.cumsum(label) for vid,label in data.label_dict.items()}
    if OC.select(cfg, 'base.path.pseudo_path'):
        data.pseudo_dict = pd.read_pickle(cfg.base.path.pseudo_path)
    if OC.select(cfg, 'base.path.filter_path'):
        data.filter_dict = pd.read_pickle(cfg.base.path.filter_path)
    if OC.select(cfg, 'base.path.info_path'):
        with open(cfg.base.path.info_path) as f:
            data.info_dict = json.load(f)
        fps_dict = {k:v['fps'] for k,v in data.info_dict.items()}
        data.duration_dict = {k:np.diff(v).flatten()/fps_dict[k] for k,v in data.shot_bound.items()}

    # ignore bad videos
    if OC.select(cfg, 'base.path.split_path'):
        with open(cfg.base.path.split_path) as f:
            split_set = json.load(f)
        bad_vids = ['tt0095016', 'tt0117951', 'tt0120755'] + ['tt0258000', 'tt0120263'] + ['tt3465916']
        # bad_vids += ['tt3465916', 'tt0079944', 'tt0072443', 'tt3808342', 'tt0363589', 'tt0120263', 'tt2076220', 'tt0209463', 'tt1602620', 'tt2258281', 'tt0091406', 'tt0826711', 'tt0088222'] # gap100
        for vid in bad_vids:
            if vid in data.anno:
                del data.anno[vid]
    else:
        bad_vids = []
        split_set = {"full":list(data.anno),"train":[], "val":[], "test":list(data.anno)}

    # collecting vids
    data.full_vids = [vid for vid in split_set['full'] if vid not in bad_vids]
    data.train_vids = [vid for vid in data.anno.keys() if vid in split_set['train']]
    data.val_vids = [vid for vid in data.anno.keys() if vid in split_set['val']]
    data.test_vids = [vid for vid in data.anno.keys() if vid in split_set['test']]
    data.other_vids = [vid for vid in data.full_vids if vid not in data.anno.keys() and vid not in bad_vids]
    data.vid2vno = {vid:vno for vno,vid in enumerate(data.full_vids)}
    data.vno2vid = {vno:vid for vno,vid in enumerate(data.full_vids)}
    data.shot_num = Munch({vid:len(data.shot_bound[vid]) for vid in data.full_vids})
    data.vno_shot_bound = {data.vid2vno[vid]:bounds for vid,bounds in data.shot_bound.items() if vid in data.vid2vno}

    # initialize the seed
    if OC.select(cfg, 'base.seed'):
        def setup_seed(seed):
            if seed is None:
                return
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        setup_seed(cfg.base.seed)
