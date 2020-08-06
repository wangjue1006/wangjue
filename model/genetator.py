import torch.nn as nn
import torch.nn.functional as F
class Generater(nn.Module):
    def __init__(self,size,d_model):
        super(Generater,self).__init__()
        self.size=size
        self.linear=nn.Linear(d_model,self.size)

    def forward(self,x):
        return F.log_softmax(self.linear(x),dim=-1)