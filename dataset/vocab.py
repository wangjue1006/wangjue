import jieba
import re
cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
class Vocab:
    def __init__(self,max_size):
        self.stoi={'<sos>':0,'<pad>':1,'<eos>':2}
        self.itos=[]
        self.first_token={}
        self.max_size=max_size
        for item in self.stoi:
            self.itos.append(item)

    def s2i(self,words):
        for word in words:
            if word not in self.stoi:
                self.stoi[word]=len(self.itos)
                self.itos.append(word)

    def i2s(self,seq):
        res=[]
        for i in seq:
            word=self.itos[int(i)]
            res.append(word)
        return ''.join(res)
