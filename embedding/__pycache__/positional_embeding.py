import torch
import torch.nn as nn
import math
from torch.autograd import Variable
class Positional_Embedding(nn.Module):
    def __init__(self,max_length,d_model,drop_out):
        super(Positional_Embedding,self).__init__()
        max_length+=10
        pe=torch.zeros(max_length,d_model)
        pos=torch.arange(0.,max_length).unsqueeze(1)
        div_term=torch.exp(torch.arange(0.,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div_term)
        pe[:,1::2]=torch.cos(pos*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        self.dropout=nn.Dropout(drop_out)

    def forward(self,x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)
