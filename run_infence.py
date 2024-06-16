# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
import os
import time
import argparse
from infence import get_infence
from infence_vis import get_infence_vis

from Config import Config
argement = argparse.ArgumentParser(description='运行参数')
argement.add_argument('--run-type', default=0, type=int, metavar='N',help='run-type default (0)')
run_args = argement.parse_args()
def get_infence_list(device, model_type, dataset, epochs, trian_time=1):
    infence_list = []
    for epoch in epochs:
        #[model_type,device,train_times, ckpt_epoch, dataset_name]
        infence_list.append([model_type, device, trian_time, epoch, dataset])
    return infence_list
if __name__ == '__main__':
           
    if  run_args.run_type == 1 :  
        d=0        
        conf = Config()
        setattr(conf, 'config', './configs/imagenet.yaml')
        setattr(conf, 'model_name', 'mobilevit_xs')
        setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        
        for mod in [45]:
            epochs = [i for i in range(160, 199, 2)]
            settings =  get_infence_list(d, mod, 'train', epochs, 1)
            for setting in settings:            
                conf.set_infence(setting)
                get_infence(conf, [1,2,3,4], 1)      
                
        
        for mod in [45]:
            epochs = [i for i in range(30, 90, 15)]
            settings =  get_infence_list(d, mod, 'train', epochs, 1)
            for setting in settings:            
                conf.set_infence(setting)
                get_infence(conf, [3], 1)              
        # for mod in [12, 13, 14, 16, 18]:
        #     epochs = [i for i in range(160, 198, 2)]
        #     settings =  get_infence_list(d, mod, 'train', epochs, 1)
        #     for setting in settings:            
        #         conf.set_infence(setting)
        #         get_infence(conf, [1,2,3,4], 1)      
        
 

        # d=1         
        # conf = Config()
        # setattr(conf, 'config', './configs/imagenet.yaml')
        # setattr(conf, 'model_name', 'mobilevit_xs')
        # setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        
        # for mod in [26]:
        #     epochs = [i for i in range(160, 198, 2)]
        #     settings =  get_infence_list(d, mod, 'train', epochs, 2)
        #     for setting in settings:            
        #         conf.set_infence(setting)
        #         get_infence(conf, [1,2,3,4], 1) 
                

