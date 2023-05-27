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
        payload['center_label'] = data.label_dict[vid][center_shid]
        payload['all_label'] = torch.tensor(data.label_dict[vid][begin_shid:end_shid])
        shids = list(range(begin_shid, end_shid))
        clip = self.load_clip(vid, shids)
        payload['video'] = torch.stack(clip)
        if data.get('duration_dict', None):
            payload['duration'] = torch.tensor(data.duration_dict[vid][begin_shid:end_shid], dtype=torch.float32)
        return payload
    
    def __len__(self):
        if cfg.base.stage=='finetune' and cfg.get('finetune', None):
            p = cfg.finetune.params.get('label_percentage', False)
            if p != False: return int(len(self.payload_list) * p/100)
        return len(self.payload_list)