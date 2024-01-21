import torch
import pandas as pd
import Config as config
device=config.device

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, path,tokenizer):
        data=pd.read_json(path,lines=True)
        self.x = data['content']
        self.y = data['summary']
        self.tokenizer=tokenizer
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.x)
    
    #对数据额外的处理，主要是对长度进行补齐
    def collate_fn(self,batch_data):
        x,y = list(zip(*batch_data))
        code_x=self.tokenizer(x,padding=True,truncation=True)
        code_y=self.tokenizer(y,padding=True,truncation=True)
        # code_x=self.tokenizer(x,padding=True)
        # code_y=self.tokenizer(y,padding=True)
        x = code_x['input_ids']
        y = code_y['input_ids']
        x_mask=code_x['attention_mask']
        y_mask=code_y['attention_mask']
        x=torch.tensor(x)#把结束符号去掉，用于训练的时候错开一个
        y=torch.tensor(y)#y原本需要一个原始的计算代价，一个去掉开始符号用于预测的答案，为了节省显存，到时候就直接在原始y上操作
        #与多头注意力的attention对齐，如果没有这个unsqueeze(1)，那么到mask的时候是batch_size,8,118,118对batch_size,1,118
        #这个地方先保留一下，为什么人家不直接阔两个维度？，难道是我在数据处理的时候那个地方少了一个括号？
        
        x_mask=torch.tensor(x_mask).unsqueeze(1)
        if(config.global_overview):#如果引入全局概述，则需要扩充一个1
            x_mask=torch.cat((torch.ones(x_mask.size(0),x_mask.size(1),1),x_mask),dim=-1)

        y_mask=torch.tensor(y_mask)[:,1:].unsqueeze(1)#本来是把最后一个词的掩码去掉，直接去前面一样，相当于都少了一个1
        tri_l=y_mask.size(-1)
        lower_triangle=torch.tril(torch.ones(tri_l,tri_l),diagonal=0).unsqueeze(0)
        y_mask=torch.logical_and(lower_triangle,y_mask)
        # print(x.size(),y.size(),x_mask.size(),y_mask.size())
        return x.to(device),y.to(device),x_mask.to(device),y_mask.to(device)
