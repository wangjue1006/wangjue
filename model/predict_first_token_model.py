import torch
import torch.nn as nn

class Predict_Fist_Token(nn.Module):
    def __init__(self,seq_len,d_model,first_token_vocab_size):
        super(Predict_Fist_Token, self).__init__()
        self.main1=nn.Sequential(
            nn.Linear(seq_len,1),
            nn.ReLU(),
        )
        self.d_model=d_model
        self.main2=nn.Sequential(
            nn.Linear(d_model,16*d_model),
            nn.ReLU(),
            nn.Linear(d_model*16,8*d_model),
            nn.ReLU(),
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),          
            nn.Linear(d_model * 8, 8 * d_model),
            nn.ReLU(),
            nn.Linear(d_model*8,d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4,first_token_vocab_size),
            nn.LogSoftmax()
        )

    def forward(self,x):
        x=x.transpose(-1,-2)
        x=self.main1(x).view(-1,self.d_model)
        return self.main2(x)