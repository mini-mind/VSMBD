# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class MlpHead(nn.Module):
    def __init__(self, clip_len):
        super().__init__()
        self.clip_len=clip_len

        self.model = nn.Sequential(
            nn.Linear(2048*clip_len, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(4096, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(.1),
            nn.Linear(1024, 2, bias=True)
        )

    def forward(self, x):
        # x shape: [b t d] where t means the number of views
        x = self.model(x.reshape([-1,2048*self.clip_len]))
        return x
        # return F.normalize(x, dim=-1)
