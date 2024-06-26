import copy

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..common.config import *
from .base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, vid_list):
        super().__init__(vid_list)
                
    def __getitem__(self, idx):

        # payload(['vid', 'vno', 'center_shid', 'shids'])
        payload = self.payload_list[idx]
        payload = copy.copy(payload)
        begin_shid = payload['begin_shid']
        del payload['begin_shid']
        center_shid = begin_shid+self.left_len
        payload['center_shid'] = center_shid
        end_shid = begin_shid+self.clip_len
        vid = payload['vid']
        vno = data.vid2vno[vid]
        payload['vno'] = vno
        payload['shids'] = torch.arange(begin_shid, end_shid)

        # payload(['pos_idx'])
        pseudo_bound = np.random.randint(self.clip_len-2)+1
        payload['pos_idx'] = pseudo_bound

        # payload(['video'])
        rand_shid = np.random.randint(data.shot_num[vid]-self.clip_len)
        shids_left = list(range(begin_shid, begin_shid+pseudo_bound))
        shids_right = list(range(rand_shid, rand_shid+(self.clip_len-pseudo_bound)))
        clips = self.load_clip(vid, shids_left) + self.load_clip(vid, shids_right)
        payload['video'] = torch.stack(clips)

        # payload(['neg_idx'])
        bounds = [0, pseudo_bound]
        left_bounds = [begin_shid-s for s in shids_left if data.filter_dict[vid].get(s, 0)>0.5]
        right_bounds = [pseudo_bound+rand_shid-s for s in shids_right if data.filter_dict[vid].get(s, 0)>0.5]
        bounds += left_bounds + right_bounds
        no_bounds = np.arange(self.clip_len)
        no_bounds = list(set(no_bounds) - set(bounds))
        if len(bounds) == 0:
            neg_idx = np.random.randint(1,self.clip_len-1)
            if neg_idx==pseudo_bound: neg_idx+=1
        else:
            neg_idx = np.random.choice(no_bounds)
        payload['neg_idx'] = neg_idx

        return payload
    
    def __len__(self):
        return len(self.payload_list)