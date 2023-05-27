from torch import nn

class MlpHead(nn.Module):
    def __init__(self, hid_dim=512, out_dim=2, bn=False):
        super().__init__()
        self.out_dim = out_dim
        self.fc1 = nn.LazyLinear(hid_dim)
        self.bn = nn.LazyBatchNorm1d() if bn else None
        self.relu = nn.ReLU()
        self.fc2 = nn.LazyLinear(out_dim)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.fc1(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.reshape(*shape[:-1], self.out_dim)
        return x