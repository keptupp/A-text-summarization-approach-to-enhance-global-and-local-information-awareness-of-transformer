#数据的读取、模型的训练预测都在这里
import Train
import Transformer_Model
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from DataLoad import GetLoader
import torch
import Config as config
import os
import utils
from transformers import PegasusTokenizer
device=config.device

#bert编码中文模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",use_fast=True,model_max_length=config.d_model)

#bert编码英文模型
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",use_fast=True,model_max_lengt=1000)

# tokenizer=PegasusTokenizer.from_pretrained("google/pegasus-xsum",use_fast=True,model_max_length=768)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",use_fast=True)

# 获取数据对象
print('读取数据中。。。')
torch_train_datas=GetLoader(config.train_file,tokenizer)
torch_dev_datas=GetLoader(config.dev_file,tokenizer)
torch_test_datas=GetLoader(config.test_file,tokenizer)
# 将数据分成batch_size并padding
train_datas = DataLoader(torch_train_datas, batch_size=32, shuffle=True, drop_last=False,collate_fn=torch_train_datas.collate_fn)
dev_datas = DataLoader(torch_dev_datas, batch_size=32, shuffle=True, drop_last=False,collate_fn=torch_dev_datas.collate_fn)
test_datas = DataLoader(torch_test_datas,batch_size=32,shuffle=True, drop_last=False,collate_fn=torch_test_datas.collate_fn)

out_text_dataset=GetLoader('/home/liuzheng/AI_Learn/A_data/test_csl.json',tokenizer)
out_text_loader=DataLoader(out_text_dataset,batch_size=32,shuffle=False, drop_last=False,collate_fn=torch_test_datas.collate_fn)

#bert的词袋大小
print('数据读取完成')
vocab_size=tokenizer.vocab_size
config.vocab_size=tokenizer.vocab_size

#定义transformer模型
model=Transformer_Model.get_model(config,src_vocab=vocab_size,trg_vocab=vocab_size).to(device)

#优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
#计算代价
lossCompute=torch.nn.CrossEntropyLoss(reduction="mean").to(device)


def run():
    #训练好的模型
    start_epoch=0
    if(os.path.exists(config.model_data)):
        print("加载权重。。。")
        checkpoint = torch.load(config.model_data)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch']
    else:
        print("没有权重，随机初始化")


    # Train.train(model,train_datas,dev_datas,optimizer,lossCompute,tokenizer,start_epoch,test_datas=test_datas)
    scores=Train.pred_epoch(model,out_text_loader,tokenizer,use_beam=False)
    print(scores)
    print((scores['rouge-1']['f']+scores['rouge-l']['f'])/2)
    # utils.write_txt(str(scores))

run()