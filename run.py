# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
import os
import time
import argparse
from Config import Config
from train import get_trianer
from train_vis import get_trianer_vis
from infence import get_infence

argement = argparse.ArgumentParser(description='运行参数')
argement.add_argument('--run-type', default=1, type=int, metavar='N',help='run-type default (0)')
#parser = argparse.ArgumentParser(description='ImageNet Evaluation', add_help=True)
#parser.add_argument('--config', type=str, default='./configs/imagenet.yaml', help="Configuration file")
#parser.add_argument('--model_name', type=str, default='mobilevit_xxs', help="Model name")
#parser.add_argument('--weights', type=str, default='./weights/mobilevit_xxs.pt', help="Model weights")

run_args = argement.parse_args()
def get_infence_list(device, model_type, dataset, epochs, trian_time=1):
    infence_list = []
    for epoch in epochs:
        #[model_type,device,train_times, ckpt_epoch, dataset_name]
        infence_list.append([model_type, device, trian_time, epoch, dataset])
    return infence_list
if __name__ == '__main__':
    if  run_args.run_type == 1:    
        
        
        d =2
#        conf = Config()
#        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
#        setting=[10, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
#        conf.set_train_conf(setting)
#        setattr(conf, 'config', './configs/imagenet.yaml')
#        setattr(conf, 'model_name', 'mobilevit_xs')
#        setattr(conf, 'weights', './weights/mobilevit_xs.pt')
#        get_trianer(conf, 1)  
        
#        conf = Config()
#        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
#        setting=[11, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
#        conf.set_train_conf(setting)
#        setattr(conf, 'config', './configs/imagenet.yaml')
#        setattr(conf, 'model_name', 'mobilevit_xs')
#        setattr(conf, 'weights', './weights/mobilevit_xs.pt')
#        get_trianer(conf, 1)  
#        
        # conf = Config()
        # #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        # setting=[13, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
        # conf.set_train_conf(setting)
        # setattr(conf, 'config', './configs/imagenet.yaml')
        # setattr(conf, 'model_name', 'mobilevit_xs')
        # setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        # get_trianer(conf, 2)  
        
            
        # for i in [ 32]:  #30,27,
        #     conf = Config()
        #     #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        #     setting=[i, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
        #     conf.set_train_conf(setting)
        #     setattr(conf, 'config', './configs/imagenet.yaml')
        #     setattr(conf, 'model_name', 'mobilevit_xs')
        #     setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        # # #     get_trianer(conf, 4)    
        # for i in range(38141, 381412):  #30,27,
        #     conf = Config()
        #     #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        #     setting=[i, 17, d, 0.002, 170, 0.8, 15, 'train', 1, 99]
        #     conf.set_train_conf(setting)
        #     conf.start_ckpt_epoch=120
        #     setattr(conf, 'config', './configs/imagenet.yaml')
        #     setattr(conf, 'model_name', 'mobilevit_xs')
        #     setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        #     get_trianer(conf, 7)    
            
        
        # conf = Config()
        # setattr(conf, 'config', './configs/imagenet.yaml')
        # setattr(conf, 'model_name', 'mobilevit_xs')
        # setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        
        # for mod in  range(38141, 381416):
        #     epochs = [i for i in range(120, 170,2)]
        #     settings =  get_infence_list(d, mod, 'train', epochs, 1)
        #     for setting in settings:            
        #         conf.set_infence(setting)
        #         get_infence(conf, [1], 1)  
                
            
        for i in [ 3814]:  #30,27,
            conf = Config()
            #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
            setting=[i, 17, d, 0.002, 200, 0.1, 100, 'train', 1, 128]
            conf.set_train_conf(setting)
            setattr(conf, 'config', './configs/imagenet.yaml')
            setattr(conf, 'model_name', 'mobilevit_xs')
            setattr(conf, 'weights', None)
            get_trianer(conf, 7)      
            

        for mod in  [ 3814]:
            epochs = [i for i in range(160, 180, 2)]
            settings =  get_infence_list(d, mod, 'train', epochs, 2)
            for setting in settings:            
                conf.set_infence(setting)
                get_infence(conf, [3], 1)  
        # conf = Config()
        # setattr(conf, 'config', './configs/imagenet.yaml')
        # setattr(conf, 'model_name', 'mobilevit_xs')
        # setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        
        # for mod in  [ 50,51, 52,53]:
        #     epochs = [i for i in range(160, 180, 2)]
        #     settings =  get_infence_list(d, mod, 'train', epochs, 1)
        #     for setting in settings:            
        #         conf.set_infence(setting)
        #         get_infence(conf, [1], 1)  
        # for i in [26]:
        #     conf = Config()
        #     #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        #     setting=[i, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
        #     conf.set_train_conf(setting)
        #     setattr(conf, 'config', './configs/imagenet.yaml')
        #     setattr(conf, 'model_name', 'mobilevit_xs')
        #     setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        #     get_trianer(conf, 2)            
        

        
#        for i in [ 21,  22, 23 ]:
#            conf = Config()
#            #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
#            setting=[i, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
#            conf.set_train_conf(setting)
#            setattr(conf, 'config', './configs/imagenet.yaml')
#            setattr(conf, 'model_name', 'mobilevit_xs')
#            setattr(conf, 'weights', './weights/mobilevit_xs.pt')
#            get_trianer(conf, 3)  
        
        
      

        
#        conf = Config()
#        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
#        setting=[16, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
#        conf.set_train_conf(setting)
#        setattr(conf, 'config', './configs/imagenet.yaml')
#        setattr(conf, 'model_name', 'mobilevit_xs')
#        setattr(conf, 'weights', './weights/mobilevit_xs.pt')
#        get_trianer(conf, 3)  
#        
#        conf = Config()
#        #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
#        setting=[18, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
#        conf.set_train_conf(setting)
#        setattr(conf, 'config', './configs/imagenet.yaml')
#        setattr(conf, 'model_name', 'mobilevit_xs')
#        setattr(conf, 'weights', './weights/mobilevit_xs.pt')
#        get_trianer(conf, 3)  
#        
        
        # conf = Config()
        # #[model_type, loss_type, device, lr,  epoch, lr_decay_rate, lr_decay_epoch, dataset_name]
        # setting=[14, 16, d, 0.002, 200, 0.8, 19, 'train', 1, 99]
        # conf.set_train_conf(setting)
        # setattr(conf, 'config', './configs/imagenet.yaml')
        # setattr(conf, 'model_name', 'mobilevit_xs')
        # setattr(conf, 'weights', './weights/mobilevit_xs.pt')
        # get_trianer(conf, 3)  
