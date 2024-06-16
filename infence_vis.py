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
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import matplotlib.image  as mpimg
#############################################
import tools.Tools as tool
import tools.SOD_loader as data_loader
from tools.metrics import Eval_tool
#############################################
from model import get_model
import Loss as ls
import Config
import os
import torch.nn.functional as F
#############################################
def save_d(img_tensor, dir_path):
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()
    out = out[0]
    out =  out[0]
    mpimg.imsave(dir_path,out, cmap='gray')
    
def save_r(img_tensor, dir_path):
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()[0]
    out = out.transpose((1, 2, 0))
    mpimg.imsave(dir_path,out)
def val_iter(net, val_data_loader, device, dir_path, eval_tool):
    net.eval()
    with torch.no_grad():    
    #######验证过程
        for i_batch, sample_batched in enumerate(val_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].cuda()
            thermal = sample_batched['thermal'].cuda()
            gt = sample_batched['label'].cuda()
            name = sample_batched['name']
#            h  = sample_batched['height']
#            w  = sample_batched['width']
            out  = net(rgb, thermal)
            b,c,h,w = rgb.shape
            save_r(rgb, dir_path +'/' + name[0] + '_rgb.png')
            save_d(thermal, dir_path +'/' + name[0] + '_d.png')
            save_d(gt, dir_path +'/' + name[0] + '_gt.png')
            for i, o in enumerate(out):
#            out0 = F.interpolate(out0, (h, w), mode='bilinear')
                o = F.interpolate(o, (h//2, w//2), mode='bilinear')
                save_d(o, dir_path +'/' + name[0] +str(i)+ '.png')
                
            if i_batch > 100:
                 break
            out0 = out[0]
            eval_tool.run_eval(out0, gt)
    ##################每个epoch的损失统计和打印#############################
    mae, maxF, Sm = eval_tool.get_score()
    return  mae, maxF, Sm    
             
def val_iter1(net, val_data_loader, device, dir_path, eval_tool):
    net.eval()
    with torch.no_grad():    
    #######验证过程
        for i_batch, sample_batched in enumerate(val_data_loader):
            ################读取数据 ################
            rgb = sample_batched['image'].cuda()
            thermal = sample_batched['thermal'].cuda()
            gt = sample_batched['label'].cuda()
            name = sample_batched['name']
            out  = net(rgb, thermal)
            out0 = out
            save_d(out0, dir_path +'/' + name[0] + '.png')
            eval_tool.run_eval(out0, gt)
            ############每个batch的结果统计和输出##########
            if i_batch % 200 == 0 and i_batch > 0:
                print('-->step:' , i_batch, 'done!')
    ##################每个epoch的损失统计和打印#############################
    mae, maxF, Sm = eval_tool.get_score()
    print('mae:', mae, 'max F measure:', maxF, 'S measure:' , Sm)
    return  mae, maxF, Sm
          
def train(args = Config.Config(), val_fun=val_iter, datasets=1):
    args.name= 'infence' 
################################准备设备########################################
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    if  args.is_cuda and torch.cuda.is_available():
        device = args.device
        
###############################################################################
################################加载模型########################################
    net = get_model(args.model_type)
    net.cuda()
    print('训练模型为: ', args.model_type, 'GPU:' ,  device)
###############################################################################
################################加载参数######################################
    _, epoch = tool.load_ckpt(net, None, args.last_ckpt)
    
    
###############################################################################
#####################读取数据###################################################
    Date_File = "/data/HNC/dataset/RGBD_SOD/"
    image_h, image_w = data_loader.get_Parameter()
    print('##########################',epoch,'################################')
    for dataset in datasets:
    ###############################################################################
    ####################读取测试数据###################################################
        val_dataset = data_loader.data_loader(Date_File,'test', subset=dataset,
                                        transform=transforms.Compose([
                                           data_loader.scaleNorm(image_w, image_h, (1, 1.2), False),
                                           data_loader.ToTensor(),
                                           ]))
        val_data_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=2)
    #####################创建目录##########################################################
        time_dir = './vis_sal_map174'
        if not os.path.exists(time_dir):
            os.makedirs(time_dir)
        eval_tool = Eval_tool()
        eval_tool.reset()
        begin_time = time.time()
        mae, maxF, Sm = val_fun(net, val_data_loader, device, time_dir, eval_tool)
        totall_time =  time.time()-begin_time
        avg_time = totall_time / len(val_data_loader)          
        print(avg_time)
        print('##########################################')
    print('##########################',epoch,' end ################################')
def get_infence_vis(args, dataset = [1,],model_tpye = 1):      
  if model_tpye == 1:
    train(args, val_iter, dataset) 
    
    
if __name__ == '__main__':
    train()