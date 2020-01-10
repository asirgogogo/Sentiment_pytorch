import torch
from torch import nn,optim
import torch.utils.data as Data
import numpy as np
import time

def generator(data,labels,batch_size,train=True):
    train_set=Data.TensorDataset(data,labels)
    train_iter=Data.DataLoader(train_set,batch_size,shuffle=train)
    return train_iter

class BiRNN(nn.Module):
    def __init__(self,word_index,embed_size,num_hiddens,num_layers):
        super(BiRNN,self).__init__()
        self.embedding=nn.Embedding(len(word_index)+1,embed_size)

        self.encoder=nn.LSTM(input_size=embed_size,
                             hidden_size=num_hiddens,
                             num_layers=num_layers,
                             bidirectional=True)
        #pytorch在使用Crossentropy的时候是不需要做one-hot，系统会自动做。
        #所以这个地方输出是2
        self.decoder=nn.Linear(4*num_hiddens,2)
    def forward(self,inputs):
        embeddings=self.embedding(inputs.permute(1,0))
        outputs,_=self.encoder(embeddings)
        encoding=torch.cat((outputs[0],outputs[-1]),-1)
        outs=self.decoder(encoding)
        return outs

def evaluate_accuracy(data_iter, net):
    acc_sum, n = torch.tensor([0]), 0
    for X,y in data_iter:
        acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train(train_iter,test_iter,net,loss,optimizer,num_epochs):
    batch_count=0
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start=0.0,0.0,0,time.time()
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
            batch_count+=1
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc %.3f,test_acc %.3f,time %.1f sec'%(epoch+1,train_l_sum/batch_count,train_acc_sum/n,test_acc,time.time()-start))