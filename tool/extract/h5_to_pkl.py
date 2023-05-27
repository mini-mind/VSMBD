from ast import parse
import os, h5py, numpy as np, pandas as pd
from tqdm import tqdm
# from argparse import ArgumentParser

from .config import cfg


source = cfg.save_path
if source.endswith('.h5'):
    target = cfg.save_path[:-2]+'pkl'
else:
    raise NotImplementedError("Can not save as this type")

with h5py.File(source, 'r') as rf:
    dataset = {}
    for vid,shots in tqdm(rf.items()):
        dataset[vid] = {}
        for shid,data in shots.items():
            dataset[vid][shid] = np.array(data)            
pd.to_pickle(dataset, target)