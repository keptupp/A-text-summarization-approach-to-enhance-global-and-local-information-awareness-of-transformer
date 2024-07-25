import torch
import torch.nn as nn
import Config as config

class CnnLayer(nn.Module):
    def __init__(self,d_model,cnn_size,step):
        super().__init__()
        self.cnn1d=nn.Conv1d(d_model,d_model,cnn_size,stride=step)
        self.pool1d=nn.MaxPool1d(cnn_size,stride=step)

    def forward(self,x):
        #注意Encoder输出的形状是(batchsize,长度,维度)，在卷积的时候需要调换长度和维度，因为默认对最后一个维度卷积
        total_overview=x[:,0,:].unsqueeze(1)#将全局的信息分开
        x=x[:,1:,:]
        x=self.pool1d(self.cnn1d(x.transpose(-1,-2))).transpose(-1,-2)
        #然后拼接复原
        x=torch.cat((total_overview,x),dim=1)
        return x







if __name__ == "__main__":
    #测试卷积网络
    m = nn.Conv1d(512, 512, 5, stride=1)
    input = torch.randn(16, 50, 512)
    output = m(input.transpose(-1,-2))
    print(output.size())
    pool=nn.MaxPool1d(3,stride=1)
    output=pool(output)
    print(output.size())
    print(input.size())