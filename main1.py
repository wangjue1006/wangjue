from embedding.embedding import Embedding
from embedding.positional_embeding import Positional_Embedding
from dataset.vocab import Vocab
from dataset.dataset import Transformer_Dataset
from torch.utils.data import DataLoader
from model.model import Encoder_Decoder
from model.encoder import Encoder,Encoder_Layer
from model.decoder import Decoder,Decoder_Layer
from model.attention import MultiHead_Attention
from model.feedward_net import feedback_net
from model.genetator import Generater
import  torch.nn as nn
import  torch.optim as Optim
import torch
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-d','--data',required=True,help='please set your dataset')
parser.add_argument('-m','--max_size',default=25,type=int,help='the sentence max length')
parser.add_argument('--batch_size',default=128,type=int,help='the batch size')
parser.add_argument('--d_model',default=64,type=int,help='the dimintion of model')
parser.add_argument('-t','--if_train',type=bool,default=False,help='train or test')
parser.add_argument('--h',default=8,type=int,help='how many heads')
parser.add_argument('--n_layers',default=16,type=int,help='how many layers')
parser.add_argument('-c','--if_cuda',default=True,type=bool,help='if need gpu')
parser.add_argument('--dropout',type=int,default=0.8,help='the dropout')
parser.add_argument('--lr',type=int,default=0.005,help='the learning rate')
parser.add_argument('--test_file',type=str,default='./bot.pkl',help='the model\'s parameters file')
args=parser.parse_args()

def read_data(file):
    text=open(file,'r')
    data=text.read().split('\nE\n')
    data = list(map(lambda x: func(x), data))
    data[0] = data[0][1:]
    return data

def func(x):
    x=x.strip().split('\n')
    x=list(map(lambda y:y[2:],x))
    return x

max_length=args.max_size
BATCH_SIZE=args.batch_size
d_model=args.d_model

def build_vocab(data):
    try:
        vocab=args.vocab
        vocab_size=len(args.vocab.itos)
    except:
        vocab=Vocab(max_length)
        for ss in data:
            for s in ss:
                vocab.s2i(s)
        vocab_size=len(vocab.itos)
    return vocab

def sub_mask(size):
    mask=(torch.triu(torch.ones(1,size,size),1)==0)
    return mask

def built_dataset(vocab):


    dataset = Transformer_Dataset(data, max_length, vocab,args.if_train)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader

def make_model(args):
    positional_encoding=Positional_Embedding(max_length,d_model,args.dropout)
    src_embedded=Embedding(vocab_size,d_model,positional_encoding)
    tgt_embedded=Embedding(vocab_size,d_model,positional_encoding)
    self_attn_encoder=MultiHead_Attention(args.h,d_model)
    self_attn_decoder=MultiHead_Attention(args.h,d_model)
    self_attn_src=MultiHead_Attention(args.h,d_model)
    feedward_net_encoder=feedback_net(d_model,args.dropout)
    feedward_net_decoder=feedback_net(d_model,args.dropout)
    encoder_layer=Encoder_Layer(d_model,self_attn_encoder,feedward_net_encoder)
    decoder_layer=Decoder_Layer(d_model,self_attn_decoder,self_attn_src,feedward_net_decoder)
    encoder=Encoder(encoder_layer,args.n_layers)
    decoder=Decoder(decoder_layer,args.n_layers)
    generator=Generater(vocab_size,d_model)
    model=Encoder_Decoder(encoder,decoder,src_embedded,tgt_embedded,generator).to(device)
    return model.to(device)

device=torch.device('cuda' if torch.cuda.is_available() and args.if_cuda  else 'cpu')
n=0
epoches=500
step=0
teacher_forceint=0.8
clip=50.0

def seq2sentence(seq):
    return vocab.i2s(seq)

def train(data_loader,model,epoches,step):
    loss_fn = nn.NLLLoss(ignore_index=vocab.stoi['<pad>']).to(device)
    optim = Optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    step_lr = Optim.lr_scheduler.MultiStepLR(optim, [1000, 5000, 10000], gamma=0.1)
    for epoch in range(epoches):
        for x, y,src_mask,tgt_mask,tgt_i in data_loader:
            tgt_i=tgt_i.to(device)
            x=x.to(device)
            y=y.to(device)
            src_mask, tgt_mask=src_mask.to(device),tgt_mask.to(device)
            
            hidden=y
            tgt_new=model(x,hidden.view(-1,max_length),src_mask,tgt_mask)
            pred = torch.argmax(tgt_new, dim=-1).long()

            loss=0

            print(seq2sentence(y[-1].view(-1,1)))
            print('pred_y:',seq2sentence(pred[-1].view(-1,1)))

            loss+=loss_fn(tgt_new.view(len(x)*max_length,-1),tgt_i.view(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optim.step()
            step_lr.step(step)
            step+=1
            print(step)
            print(loss,epoch)
            if step%1000==0:
                torch.save(model.state_dict(), './bot.pickle')

def talk(data_loader,model,file):
    model.load_state_dict(torch.load(file))
    for x,src_mask in data_loader:
        red=[]
        tgt = torch.full((len(x), 1), vocab.stoi['<sos>']).long().to(device)
        for i in range(vocab.max_size):
            x=x.to(device).long()
            src_mask=src_mask.to(device).long()
            print('tgt:',tgt)
            tgt_new=model(x,tgt,src_mask,sub_mask(tgt.size(-1)).to(device).long())
            tgt_new=tgt_new[:,-1]
            pred = torch.argmax(tgt_new[:,1:], dim=-1).long().view(-1,1)
            red.append(pred[-1])
            tgt = torch.cat((tgt, pred), dim=-1)
            if pred.view(-1)==vocab.stoi['<eos>']:
                break


        print('pred:',seq2sentence(red))
data=read_data(args.data)
vocab=build_vocab(data)
vocab_size=len(vocab.itos)
data_loader=built_dataset(vocab)
model=make_model(args)

if args.if_train:
    train(data_loader,model,epoches,step)
else:
    talk(data_loader,model,args.test_file)


