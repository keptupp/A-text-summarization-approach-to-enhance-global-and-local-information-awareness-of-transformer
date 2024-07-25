#用于生成全局注意力的编码片段
import torch
import torch.nn as nn
import Config as config

class Lstm(nn.Module):
    def __init__(self,feature_len,hidden_len,num_layers):
        super().__init__()
        self.lstm=nn.LSTM(feature_len,hidden_len,num_layers,batch_first=True)#分别表示输入的特征维度，扩展的特征维度，lstm层数
    def forward(self,x):
        return self.lstm(x)


if __name__=='__main__':
    lstm=Lstm(10,20,2)
    input=torch.randn(5,3,10)#句子长度5，batchsize为3，特征数量10
    #不提供h0c0则默认为0
    # h0=torch.randn(2,3,20)
    # c0=torch.randn(2,3,20)
    output,(hn,cn)=lstm(input)
    print(output.size())#[5,3,20]

