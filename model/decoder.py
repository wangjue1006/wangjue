import torch.nn as nn
from utils.clones import Clones
from .subconnection import Suberconnection
clones=Clones()
class Decoder_Layer(nn.Module):
    def __init__(self,d_model,self_attn,src_attn,feedback_net):
        super(Decoder_Layer,self).__init__()
        self.sublayer = clones(Suberconnection(d_model), 3)
        self.self_attn = self_attn
        self.src_attn=src_attn
        self.feedback_net = feedback_net
        self.d_model=d_model
    def forward(self,x,memory,src_mask,tgt_mask):
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer[1](x,lambda x:self.src_attn(x,memory,memory,src_mask))
        return self.sublayer[2](x,self.feedback_net)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)