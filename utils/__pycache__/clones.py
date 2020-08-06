import torch.nn as nn
class Clones(nn.Module):
    def __init__(self):
        super(Clones,self).__init__()
        pass
    def forward(self,layer,n):
        return nn.ModuleList([layer for i in range(n)])