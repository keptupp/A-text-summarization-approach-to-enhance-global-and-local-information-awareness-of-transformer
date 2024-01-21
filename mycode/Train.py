import torch
import tqdm
import Config as config
from rouge_chinese import Rouge

# from rouge import Rouge
import utils

def train_dev_epoch(model,datas,optimizer,lossCompute,tokenizer,type,epoch):
    if(type):
        model.train()
    else:
        model.eval()
    i=0
    total_loss=0
    bar=tqdm.tqdm(datas,ncols=80)
    for one in bar:
        optimizer.zero_grad()#梯度置0
        x,y,x_mask,y_mask=one
        y_pre=model(x,y[:,:-1],x_mask,y_mask).transpose(-1,-2)
        loss=lossCompute(y_pre,y[:,1:])
        if(type):#训练的时候
            loss.backward()#反向传播
            optimizer.step()#执行梯度下降
        i+=1
        total_loss+=loss.item()
        bar.set_postfix(loss=loss.item())
        if(i==len(bar)-1):#取出倒数第二个的结果
            y_pre=y_pre.transpose(-1,-2)
            y_pre=torch.argmax(torch.softmax(y_pre,dim=-1),-1)
            with open(config.summarization_file,'a',encoding='utf-8') as fp:
                fp.write("======================================第"+str(epoch)+'个epoch'+('训练' if type else '验证')+'平均代价'+str(total_loss/i)+'\n')
                for j in range(y.size(0)):
                    tgt_text='答案: '+tokenizer.decode(y[j,1:])+'\n'
                    out_text='预测: '+tokenizer.decode(y_pre[j])+'\n'+'\n'
                    fp.write(tgt_text+out_text)
    
    return total_loss/len(bar)

def pred_epoch(model,datas,tokenizer,use_beam=False):#用于推理预测，计算rouge评分
    model.eval()#表示预测，不计算梯度信息
    with torch.no_grad():
        bar=tqdm.tqdm(datas,ncols=80)
        i=0
        y_list=[]
        y_pre_list=[]
        rouge=Rouge()
        for one in bar:
            x,y,x_mask,_=one
            y_pre=model.prediction(x,x_mask,use_beam)
            i+=1
            for j in range(x.size(0)):
                if(tokenizer.decode(y[j,1:],skip_special_tokens=True)=="" or tokenizer.decode(y_pre[j],skip_special_tokens=True)==""):#验证集上有空数据，临时判断一下
                    # print('为空，预测的是',tokenizer.decode(y_pre[j],skip_special_tokens=True))
                    pass
                else:
                    y_list.append(tokenizer.decode(y[j,1:],skip_special_tokens=True))
                    y_pre_list.append(tokenizer.decode(y_pre[j],skip_special_tokens=True))
                    
        # for i in range(10):
        #     print('答案',y_list[i])
        #     print('预测',y_pre_list[i]) 
        #     print("")
        for i in range(len(y_list)):
            if y_list[i]=="测 试":
                print("测试输出",y_pre_list[i])
    return rouge.get_scores(y_pre_list, y_list,avg=True)

def train(model,train_datas,dev_datas,optimizer,lossCompute,tokenizer,start_epoch,test_datas=None):
    print('===================训练==================')
    results_curve={'train_loss':[],'dev_loss':[],'test_rouge':[]}
    early_stop=0
    best_mean_rouge=0.15
    for i in range(start_epoch,100):
        loss=train_dev_epoch(model,train_datas,optimizer,lossCompute,tokenizer,1,i+1)
        dev_loss=train_dev_epoch(model,dev_datas,optimizer,lossCompute,tokenizer,0,i+1)
        scores=pred_epoch(model,test_datas,tokenizer,use_beam=False)

        mean_rouge=(scores['rouge-1']['f']+scores['rouge-l']['f'])/2

        results_curve['train_loss'].append(loss)
        results_curve['dev_loss'].append(dev_loss)
        results_curve['test_rouge'].append(mean_rouge)
        print(i+1,'epoch，训练集平均代价：',loss,'。验证集平均代价：',dev_loss)
        print(scores)
        utils.write_log(str(i+1)+'epoch，训练集平均代价：'+str(loss)+'验证集平均代价：'+str(dev_loss))
        utils.write_log(str(scores))
        #保存模型和优化器，方便断点训练
        if(best_mean_rouge<=mean_rouge):#说明rouge评分还在上升
            best_mean_rouge=mean_rouge
            checkpoint = {
                "model": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": i+1
            }
            torch.save(checkpoint , config.model_data)
            early_stop=0
        else:#说明dev没有下降了
            early_stop+=1
            print("第",i+1,"次不降")
            utils.write_log("第"+str(i+1)+"次不降")
            if(early_stop==5):#设置五次都不下降，则视为模型已经学到头了
                print('提前结束')
                utils.write_log("提前结束")
                break
    utils.write_loss(results_curve)


