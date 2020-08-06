from torch.utils.data import Dataset
import torch
import jieba
import re
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")

def sub_mask(size):
    mask=(torch.triu(torch.ones(1,size,size),1)==0)
    return mask.unsqueeze(0)

class Transformer_Dataset(Dataset):
    def __init__(self,dataset,max_length,vocab,if_train):
        self.dataset=dataset
        self.vocab=vocab
        self.if_train=if_train
        self.max_length=max_length
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        input_=jieba.lcut(cop.sub('',self.dataset[item][0]))
        label=jieba.lcut(cop.sub('',self.dataset[item][1])) if self.if_train else None

        input_i=[]
        label_i=[]
        if self.if_train:
            for word in label:
                label_i.append(self.vocab.stoi[word])
            tgt_i =label_i+[self.vocab.stoi['<eos>']]
            label_i = [self.vocab.stoi['<sos>']] + label_i

            if len(label_i)<self.max_length:
                label_i+=[self.vocab.stoi['<pad>'] for i in range(self.max_length-len(label_i))]
                tgt_i+=[self.vocab.stoi['<pad>'] for i in range(self.max_length-len(tgt_i))]
            if len(label_i)>self.max_length:
                label_i=label_i[:self.max_length]
                tgt_i=tgt_i[:self.max_length]
            label_i = torch.LongTensor(label_i)
            tgt_mask = (label_i != self.vocab.stoi['<pad>']) & sub_mask(label_i.size(-1))


        for word in input_:
            input_i.append(self.vocab.stoi[word])
        if len(input_i)<self.max_length:
            input_i+=[self.vocab.stoi['<pad>'] for i in range(self.max_length-len(input_i))]
        if len(input_i)>self.max_length:
            input_i=input_i[:self.max_length]

        input_i=torch.LongTensor(input_i)

        src_mask=(input_i!=self.vocab.stoi['<pad>']).unsqueeze(-2)

        if self.if_train:
            return input_i,label_i,src_mask.unsqueeze(1),tgt_mask.squeeze(1),torch.LongTensor(tgt_i)
        else:
            return input_i,src_mask.unsqueeze(1)
