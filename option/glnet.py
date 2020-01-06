###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2018
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=2, help='segmentation classes')
        parser.add_argument('--dataset', type=str, default='DeepGlobe', help='dataset class')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--train_csv', type=str, help='csv file stores list of training images')
        parser.add_argument('--val_csv', type=str, help='csv file stores list of validation images')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3], help='mode for training procedure. 1: train global branch only. 2: train local branch with fixed global branch. 3: train global branch with fixed local branch')
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--test', action='store_true', default=False, help='test only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--sub_batch_size', type=int, default=6, help='batch size for using local image patches')
        parser.add_argument('--size_g', type=int, default=508, help='size (in pixel) for downsampled global image')
        parser.add_argument('--size_p', type=int, default=508, help='size (in pixel) for cropped local image')
        parser.add_argument('--path_g', type=str, default="", help='name for global model path')
        parser.add_argument('--path_g2l', type=str, default="", help='name for local from global model path')
        parser.add_argument('--path_l2g', type=str, default="", help='name for global from local model path')
        parser.add_argument('--lamb_fmreg', type=float, default=0.15, help='loss weight feature map regularization')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
        parser.add_argument('--device', default='cuda', help='device')
        
        # For distributed
        parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
        parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
        
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr
        if args.mode == 1 or args.mode == 3:
            args.num_epochs = 120
            args.lr = 5e-5
        else:
            args.num_epochs = 50
            args.lr = 2e-5
        return args
