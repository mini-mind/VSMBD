from torch import nn
from torchvision import models


class VisualEncoder(nn.Module):
    def __init__(self, load=False):
        super().__init__()
        model = models.resnet50(load)
        model.fc = nn.Identity()
        self.model = model
        
    def forward(self, data):
        # b, n, c, h, w = data.shape
        shape = data.shape
        data = self.model(data.view(-1, *shape[-3:]))
        return data.reshape(*shape[:-3], -1) # [b n d]
