import torch.nn as nn
import torch.nn.functional as F
class feedback_net(nn.Module):
    def __init__(self,d_model,dropout):
        super(feedback_net,self).__init__()
        self.w1=nn.Linear(d_model,d_model*10)
        self.w2=nn.Linear(d_model*10,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))
