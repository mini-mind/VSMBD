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

        # payload(['pos_idx', 'neg_idx'])
        pseudo_bound = data.pseudo_dict[vid][begin_shid]
        payload['pos_idx'] = pseudo_bound
        neg_idx = np.random.randint(self.clip_len-1)
        if neg_idx==pseudo_bound: neg_idx+=1
        payload['neg_idx'] = neg_idx

        # payload(['video'])
        shid_list = list(range(begin_shid, end_shid))
        video = self.load_clip(vid, shid_list)
        payload['video'] = torch.stack(video)
        return payload
    
    def __len__(self):
        return len(self.payload_list)