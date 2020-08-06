import torch.nn as nn

class Suberconnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        super(Suberconnection,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(size)

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))