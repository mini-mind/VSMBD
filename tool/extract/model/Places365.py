from torch import nn
from torchvision import models


class Model(nn.Module):
    def __init__(self, parameters=None):
        super().__init__()
        model = models.resnet50(False)
        if parameters is not None:
            model.fc = nn.Linear(2048, 365)
            parameters = {str.replace(k,'module.',''): v for k,v in parameters.items()}
            model.load_state_dict(parameters)
            model.fc = nn.Identity()
        self.model = model
        
    def forward(self, data):
        # b, c, h, w = data.shape
        data = self.model(data) # [b d]
        return data