
import os
import time
import sys
import torch
import yaml
import argparse
import torch.nn as nn
from detection.model_ctc import *
from detection.ctcDecoder import GreedyDecoder, BeamDecoder
from detection.data_loader import Vocab, SpeechDataset, SpeechDataLoader


# 载入参数
class Config(object):
    batch_size = 1  # 每次处理的音频数目
    dropout = 0.1

def lode_config(config_file='ctc_config.yaml'):
    '''
        输入:配置文件地址
        输出:配置类Config,内容见上
    '''
    parser = argparse.ArgumentParser()
    # parser.add_argument('--conf', help='conf file for training')
    config_file ='./detection/ctc_config.yaml'
    try:
        conf = yaml.safe_load(open(config_file ,'r' ,encoding='utf-8'))
    except:
        print("Config file not exist!")
        sys.exit(1)    

    opts = Config()
    for k ,v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))
    return opts

def load_model(opts ,model_path="./detection/ctc_best_model_attention_aug.pkl"):
    '''
        从checkpoint载入模型
        输入:
            配置类:opts
            checkpoint路径
        输出:
            载入参数后模型:model,
            输出解码器:decoder,
            单词表:vocab,
            模型存储地址:device
    '''
   
    # model_path="ctc_best_model_attention_aug.pkl"
    package = torch.load(model_path)
    # 输入243
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    # 43
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    # 81
    n_feats = package['epoch']['n_feats']
    drop_out = package['_drop_out']
    mel = opts.mel

   
    
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    use_cuda = opts.use_gpu
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    model.to(device)
    model.load_state_dict(package['state_dict'])
    model.eval()

    
    
    return model ,device

def load_data(opts ,vocab):
    '''
        从dataloader载入存储的数据
        输入:
            配置类opts,
            单词表vocab
        输出:  
            数据载入器:test_loader
    '''
    # vocab_file = opts.vocab_file
    # vocab = Vocab(vocab_file)
    test_dataset = SpeechDataset(vocab, opts.test_scp_path, opts.test_lab_path ,opts.test_trans_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=False)
    return test_loader



def detection(test_loader ,model ,decoder ,device):
    '''
        输入:
            数据载入器:test_loader
            载入后参数后模型:model
            输出解码器:decoder
            模型存储地址:device
        输出:
            检测音频中的音素结果:decodes
    '''
    decode_seq =[]  # 解码后的序列
    # human_seq=[]
    # w1 = open("decode_seq",'w+')
    # w2 = open("human_seq",'w+') 
    # total_wer = 0
    # total_cer = 0
    # device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    print("here....Start  !!!")
    start = time.time()
    # print(len(test_loader))1
    with torch.no_grad():
        decodes = []
        for data in test_loader: #
            print(111111)
            # torch.cuda.empty_cache()  #清空显存
            # target和trans一致
            # 这一句就看如何写dataloader了
            # print(data)
            inputs, input_sizes ,_ ,_, trans, trans_sizes, utt_list = data
            inputs = inputs.to(device)  # 1，156，243
            trans = trans.to(device)  # 30
            # rnput_sizes = input_sizes.to(device)
            # target = target.to(device)
            # target_sizes = target_sizes.to(device)
            
            probs = model(inputs ,trans)
            # 78，1，43
            max_length = probs.size(0)
            input_sizes = (input_sizes * max_length).long()

            probs = probs.cpu()
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())
            
            # ['sil ay s sil p ow s y uw w aa n d ah w ay sh iy ih s sil k ah m ih ng ah p hh iy ah sil']
                
                
            ## compute with out sil     
            decoded_nosil = []
            for i in range(len(decoded)):
                hyp = decoded[i].split(" ")
                hyp_precess = [ i   for i in hyp if(i != "sil")  ]
                
                decoded_nosil.append(' '.join(hyp_precess))   

            for x in range(len(decoded)):
                temp =[]
                temp.append(utt_list[x])
                temp.append(decoded_nosil[x])
                decodes.append(temp)
    print("end")
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_loader), time_used))
    return decodes


# if __name__ == "__main__":
#     '''
#         目前需要的数据文件(地址在配置文件中配置)
#         transcript_phn_text:标注音频中音素序列(后序大概率不需要)
#         fbank.scp:mel频谱索引文件(可以看一下内容,这两个使用liberosa可以提到)
#         fbank.ark:mel频谱存放文件
#         lm_phone_bg.arpa:暂时不知
#         wrd_text:文字序列
#         phn_text:文字对应的音素序列
#     '''
#     #配置载入
#     opts=lode_config(config_file='ctc_config.yaml')
#     #模型载入
#     model,decoder,vocab,device=load_model(opts,model_path="ctc_best_model_attention_aug.pkl")
#     #数据载入
#     test_loader=load_data(opts,vocab)
#     #检测结果
#     decodes=detection(test_loader,model,decoder,device)
#     print("模型检测结果为")
#     for x in decodes:
#         print(x)