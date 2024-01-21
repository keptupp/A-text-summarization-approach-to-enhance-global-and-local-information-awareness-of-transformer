import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy
import Config as config
import copy
device=config.device


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model, is_encoder=True)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model, is_encoder=False)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
    def prediction(self, src, src_mask,use_beam):
        batch_size=src.size(0)
        trg=torch.full((batch_size,1),config.cls_token_id).to(config.device)#初始化预测
        trg_mask=torch.full((batch_size,1,1),1).to(config.device)#初始化预测掩码
        #这里也一个batch的进行计算
        e_outputs = self.encoder(src, src_mask)
        len=100
        list_out=[0 for i in range(src.size(0))]
        if(not use_beam):
            for i in range(len):
                d_output_temp = self.decoder(trg, e_outputs, src_mask, trg_mask)
                output_temp=torch.argmax(torch.softmax(self.out(d_output_temp),-1),-1)
                trg=torch.cat((trg,output_temp[:,-1].unsqueeze(-1)),1)
                trg_mask=torch.tril(torch.ones(i+2,i+2),diagonal=0).unsqueeze(0).to(device)
            for i in range(trg.size(0)):
                for j in range(len-1,0,-1):
                    if(trg[i,j]==config.sep_token_id):
                        list_out[i]=trg[i,1:j]
                        break
        else:
            beam_list=torch.full((batch_size,1,config.beam_size),config.cls_token_id).to(device)#存储每种可能
            scores=torch.ones((batch_size,config.beam_size)).to(device)#存储每种可能的分数
            scores_temp=torch.zeros((batch_size,config.beam_size*config.beam_size)).to(device)#暂存比较好的分数
            out_temp=torch.zeros((batch_size,config.beam_size*config.beam_size),dtype=torch.int64).to(device)
            for i in range(len):
                if(i==0):#第一次的话，只有一个开始结果,不用循环
                    d_output_temp = self.decoder(trg, e_outputs, src_mask, trg_mask)
                    output_temp,indexs=torch.topk(torch.softmax(self.out(d_output_temp),-1),k=config.beam_size,dim=-1)
                    beam_list=torch.cat((beam_list,indexs),dim=1)#存储结果
                    scores=output_temp[:,0,:]#只保留最后的结果，不需要过程
                else:
                    for j in range(config.beam_size):#计算出j情况下的可能性                   
                        #不是第一个预测，也就是说现在beam_list里面有beam_size个结果，scores里面有给自对应的分数
                        d_output_temp = self.decoder(beam_list[:,:,j], e_outputs, src_mask, trg_mask)
                        output_temp,indexs=torch.topk(torch.softmax(self.out(d_output_temp),-1),k=config.beam_size,dim=-1)
                        #这个地方还要改一下，改成ln之后的相加，然后再去除他们的长度就比较合理
                        scores_temp[:,j:j+config.beam_size]=scores[:,j].unsqueeze(-1)*output_temp[:,-1,:]#输出的最后一个词的分数
                        out_temp[:,j:j+config.beam_size]=indexs[:,-1,:]
                    #此时根据得分选择得分最高前三个做为下次预测的beam_list
                    #这里应该做一个判断，当预测到结束符号了，就不再更新他的scores了，因为后面肯大概率是0，会降低其他预测的概率
                    one_word,one_indexs=torch.topk(scores_temp/(i+1),k=config.beam_size,dim=-1)#挑选出平均得分较高的
                    #更新分数和beam_list
                    scores=torch.gather(scores_temp,1,one_indexs)#新方法，按照指定维度，和下标去收集数据，生成新tensor
                    beam_list=torch.cat((beam_list,torch.gather(out_temp,1,one_indexs).unsqueeze(1)),dim=1)
                trg_mask=torch.tril(torch.ones(i+2,i+2),diagonal=0).unsqueeze(0).to(device)
                #测试优化，好像在循环创建的张量是不会覆盖的，所以得手动删除
                del d_output_temp,output_temp,indexs
            del scores_temp,out_temp
            #先不做截断，看看输出的内容是不是对的
            for i in range(batch_size):
                list_out[i]=beam_list[i,:,0]#因为是排序过的，所以0应该就是得分最好的
            # print(scores[:3,:])
        return list_out

            



def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model
    