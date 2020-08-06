import torch.nn as nn
from .subconnection import Suberconnection
from utils.clones import Clones
clones=Clones()
class Encoder_Layer(nn.Module):
    def __init__(self,d_model,self_attn,feedback_net):
        super(Encoder_Layer,self).__init__()
        self.sublayer=clones(Suberconnection(d_model),2)
        self.attn=self_attn
        self.d_model=d_model
        self.feedback_net=feedback_net

    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x:self.attn(x,x,x,mask))
        return self.sublayer[1](x,self.feedback_net)

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers=clones(layer,N)
        self.norm=nn.LayerNorm(layer.d_model)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)