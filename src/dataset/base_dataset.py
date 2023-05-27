import numpy as np
import torch
from tqdm import tqdm
import copy
import h5py, pandas as pd
from ..common.config import *


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, vid_list):
        super().__init__()
        self.clip_len = cfg.base.params.clip_len
        self.left_len = cfg.base.params.clip_len // 2
        self.right_len = cfg.base.params.clip_len - self.left_len
        self.payload_list = []
        self.vid_list = vid_list
        self.datasets = [self.load_dataset(path) for path in cfg.base.path.feature_path]
        for vid in tqdm(vid_list, 'Loading'):
            shot_num = data.shot_num[vid]
            for s in range(0, shot_num-self.clip_len, 1):
                self.payload_list.append({
                    'vid': vid,
                    'begin_shid': s,
                })

    def load_dataset(self, path):
        if path.endswith('.h5'):
            dataset = h5py.File(path)
        elif path.endswith('.pkl'):
            dataset = pd.read_pickle(path)
        return dataset
                
    def __getitem__(self, idx):
        payload = self.payload_list[idx]
        return payload
    
    def __len__(self):
        return len(self.payload_list)

    def load_clip(self, vid, shids):
        clip = []
        data_list = [dataset[vid] for dataset in self.datasets]
        for shid in shids:
            vec = []
            for data in data_list:
                vec.append(torch.from_numpy(np.array(data[f'{shid:04d}'])))
            vec = torch.concat(vec, dim=-1)
            clip.append(vec)
        return clip