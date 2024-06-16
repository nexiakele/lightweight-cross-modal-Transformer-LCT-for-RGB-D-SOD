# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:01:05 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:46:16 2018

@author: Dell
"""
#############################################
import time
import numpy as np
#############################################
import torch
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
#############################################
import SOD_loader as data_loader
#############################################
from torch import nn
import numpy as np
from model import get_model
import os
import matplotlib.image  as mpimg

def save_d(img_tensor, dir_path):
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()
    out = out[0]
    out =  out[0]
    mpimg.imsave(dir_path,out, cmap='gray')

def speed(net, number=1000):

    ###############################################################################
    ####################读取测试数据###################################################
    Date_File = "/data/HNC/dataset/RGBD_SOD/"
    val_dataset = data_loader.data_loader(Date_File,'test', subset=1,
                                    transform=transforms.Compose([
                                       data_loader.scaleNorm(352, 352, (1, 1.2), False),
                                       data_loader.ToTensor(),
                                       ]))
    val_data_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=2)
    #####################创建目录##########################################################
    net.eval()
    print('------------------Speed Cal----------------------')
    begin_time = time.time()
    with torch.no_grad():  
        for i_batch, sample_batched in enumerate(val_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].cuda()
            thermal = sample_batched['thermal'].cuda()
            out  = net(rgb, thermal)
            if i_batch >= number:
                break

    totall_time =  time.time()-begin_time
    avg_time = totall_time / number          
    FPS = 1 / avg_time
    return FPS


def get_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    msize = k/1e6
    return msize  
         
if __name__ == '__main__':

###############################################################################
################################加载模型########################################
    device = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    if  torch.cuda.is_available():
        device = device
    from Config import Config
    conf = Config()
    #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
    setting=[38, 16, 1, 0.002, 200, 0.8, 19, 'train', 1, 99]
    conf.set_train_conf(setting)
    setattr(conf, 'config', './configs/imagenet.yaml')
    setattr(conf, 'model_name', 'mobilevit_xs')
    setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        
    net = get_model(conf)
    
    
    net.cuda()
    FPS = speed(net)
    params = get_parameters(net)
    print(params, FPS)
    
    ######################################

