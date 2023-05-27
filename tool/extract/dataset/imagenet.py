import torch
import copy
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as T
from ..config import cfg


class Dataset(torch.utils.data.Dataset):
    def __init__(self, vid_list):
        self.payload_list = []
        for vid in tqdm(vid_list, 'Building dataset'):
            vid_root = os.path.join(cfg.keyframes_root, vid)
            files = sorted(os.listdir(vid_root))
            for shid in range(len(files)//3):
                payload = {
                    'vid': vid,
                    'shid': shid,
                }
                self.payload_list.append(payload)
        self.pipeline = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def __getitem__(self, idx):
        payload = copy.copy(self.payload_list[idx])
        names = [f'shot_{payload["shid"]:04d}_img_{k}.jpg' for k in range(3)]
        paths = [os.path.join(cfg.keyframes_root, payload['vid'], name) for name in names]
        datas = [self.pipeline(Image.open(path)) for path in paths]
        payload['data'] = torch.stack(datas)
        return payload
    
    def __len__(self):
        return len(self.payload_list)