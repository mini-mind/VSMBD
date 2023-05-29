import copy

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as T

from ..common.config import *
from .base_dataset import BaseDataset


class Dataset(BaseDataset):
    def __init__(self, vid_list):
        super().__init__(vid_list)
        print(float(cfg.pretrain.params.get('overlap', 0)))
                
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
        pseudo_bound = np.random.randint(self.clip_len-2)+1
        payload['pos_idx'] = pseudo_bound
        neg_idx = np.random.randint(1,self.clip_len-1)
        if neg_idx==pseudo_bound: neg_idx+=1
        payload['neg_idx'] = neg_idx

        # payload(['video'])
        overlap = float(cfg.pretrain.params.get('overlap', 0))
        if np.random.rand() < overlap/100:
            left_border=begin_shid-(self.clip_len-pseudo_bound)+1
            right_border=begin_shid+pseudo_bound-1
            left_border = max(left_border, 0)
            right_border = min(right_border, data.shot_num[vid]-(self.clip_len-pseudo_bound))

            # if(right_border-left_border<=0):
            #     print(left_border,right_border,':',begin_shid-(self.clip_len-pseudo_bound)+1,begin_shid+pseudo_bound-1)
            #     print(data.shot_num[vid], self.clip_len-pseudo_bound)
            #     print(begin_shid)

            rand_shid = np.random.randint(right_border-left_border+1)+left_border
        else:
            gap = cfg.pretrain.params.get('gap', 0)
            rand_shid = np.random.randint(data.shot_num[vid]-2*(gap+self.clip_len))
            if rand_shid >= begin_shid-gap-(self.clip_len-pseudo_bound):
                rand_shid += 2*gap+self.clip_len
        # rand_shid = np.random.randint(data.shot_num[vid]-self.clip_len)
        shids_left = list(range(begin_shid, begin_shid+pseudo_bound))
        shids_right = list(range(rand_shid, rand_shid+(self.clip_len-pseudo_bound)))
        clips = self.load_clip(vid, shids_left) + self.load_clip(vid, shids_right)
        payload['video'] = torch.stack(clips)
        return payload
    
    def __len__(self):
        if cfg.base.stage=='pretrain' and cfg.get('pretrain', None):
            p = cfg.pretrain.params.get('label_percentage', False)
            if p != False: return int(len(self.payload_list) * p/100)
        return len(self.payload_list)