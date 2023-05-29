import numpy as np
import torch
from tqdm import tqdm
import copy
import h5py
from .base_dataset import BaseDataset
from ..common.config import *


class Dataset(BaseDataset):
    def __init__(self, vid_list):
        super().__init__(vid_list)
                
    def __getitem__(self, idx):
        # meta-data
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
        payload['shids'] = torch.arange(begin_shid, end_shid) # 左闭右开
        # labels (whether begin of a scene)
        payload['center_label'] = 0
        shids = list(range(begin_shid, end_shid))
        clip = self.load_clip(vid, shids)
        payload['video'] = torch.stack(clip)
        return payload
    
    def __len__(self):
        return len(self.payload_list)