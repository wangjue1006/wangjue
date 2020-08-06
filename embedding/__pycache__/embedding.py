import torch.nn as nn
import math
import torch
class Embedding(nn.Module):
    def __init__(self,size,d_model,positional_embedding):
        super(Embedding,self).__init__()
        self.d_model=d_model
        self.embedding=nn.Embedding(size,d_model)
        self.positional_embedding=positional_embedding
    def forward(self,x):
        with torch.no_grad():
            x=self.embedding(x)
        return self.positional_embedding(x)*math.sqrt(self.d_model)

