import torch
model_vision='model13_80_10_10'
language="chinese"
#基本参数
d_model=512#模型词维度
n_layers=6#模型编码解码器层数
heads=8#多头注意力数量
dropout=0.1#dropout比例
load_weights=None#是否有预加载权重
device=1#指定使用的GPU
model_data='/home/liuzheng/AI_Learn/my_code/models/'+model_vision+'.pth'
summarization_file='/home/liuzheng/AI_Learn/my_code/summarization_text/'+model_vision+'.txt'
log_file='/home/liuzheng/AI_Learn/my_code/summarization_text/'+model_vision+'_log.txt'
train_loss='/home/liuzheng/AI_Learn/my_code02/summarization_text/'+model_vision+'_loss.json'
cls_token_id=101
sep_token_id=102
device = torch.device("cuda")
beam_size=3

#LCSTS_new数据集，只有训练和验证，我从验证里面抽取了前200用作测试
# _dev_file='/home/liuzheng/AI_Learn/A_data/LCSTS_new/dev.json'
# _test_file='/home/liuzheng/AI_Learn/A_data/LCSTS_new/test.json'
# _train_file='/home/liuzheng/AI_Learn/A_data/LCSTS_new/train.json'

#LCSTS正真的数据集
# dev_file='/home/liuzheng/AI_Learn/A_data/LCSTS/dev.json'
# test_file='/home/liuzheng/AI_Learn/A_data/LCSTS/test.json'
# train_file='/home/liuzheng/AI_Learn/A_data/LCSTS/train.json'

#CNN_dailymail数据集
# dev_file='/home/liuzheng/AI_Learn/A_data/CNN_dailymail/dev.json'
# test_file='/home/liuzheng/AI_Learn/A_data/CNN_dailymail/test.json'
# train_file='/home/liuzheng/AI_Learn/A_data/CNN_dailymail/train.json'

#XSUM数据集
# dev_file='/home/liuzheng/AI_Learn/A_data/xsum/valid.json'
# test_file='/home/liuzheng/AI_Learn/A_data/xsum/test.json'
# train_file='/home/liuzheng/AI_Learn/A_data/xsum/train.json'

#CNewSum数据集
# dev_file='/home/liuzheng/AI_Learn/A_data/CNewSum/dev.json'
# test_file='/home/liuzheng/AI_Learn/A_data/CNewSum/test.json'
# train_file='/home/liuzheng/AI_Learn/A_data/CNewSum/train.json'

#CSL数据集
dev_file='/home/liuzheng/AI_Learn/A_data/CSL/val.json'
test_file='/home/liuzheng/AI_Learn/A_data/CSL/test.json'
train_file='/home/liuzheng/AI_Learn/A_data/CSL/train.json'


#优化结构参数

#改动为两个地方，一个在Embed上，词袋大小转化为编码维度后，输入进LSTM中，输出最后一个维度信息，添加到transformer的输入中。
#一个在DataLoad上，若为真，则在生成文档掩码的时候多增加一个1，这个1就是新加的全局概要
global_overview=False
# vocab_size=0#用于初始化lstm,这个不用了，有一个总的embed，所以lstm这里只需要d_model到d_model
lstm_layers=5#用于设置lstm的层数
