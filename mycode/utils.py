import Config as config
import json

def write_txt(text):
    with open(config.summarization_file,'a',encoding='utf-8') as fp:
        fp.write(text+'\n')

def write_log(text):
    with open(config.log_file,'a',encoding='utf-8') as fp:
        fp.write(text+'\n')

def write_loss(text):
    with open(config.train_loss,'a',encoding='utf-8') as fp:
        fp.write(json.dumps(text))