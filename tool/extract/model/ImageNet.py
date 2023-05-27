from torch import nn
from torchvision import models


class Model(nn.Module):
    def __init__(self, parameters=None):
        super().__init__()
        model = models.resnet50(True)
        model.fc = nn.Identity()
        self.model = model
        
    def forward(self, data):
        # b, c, h, w = data.shape
        data = self.model(data) # [b d]
        return data