import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Inspector Net')
        # For dataset 
        parser.add_argument('--dataset_name', type=str, default='DeepGlobe', help='dataset class')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        
        # For model
        parser.add_argument('--n_class', type=int, default=2, help='segmentation classes')
        parser.add_argument('--backbone', type=str, default='resnet_fpn', help='backbone network')
        parser.add_argument('--refinement', type=int, default=1, help='local refinement network')
        parser.add_argument('--glob2local', action='store_true', default=False, help='Pooling features from global to local')
        parser.add_argument('--warping', action='store_true', default=False, help='Using tanh warping for local branch')
        parser.add_argument('--num_scaling_level', type=int, default=3, help='number of scaling level')
        parser.add_argument('--restore_path', type=str, default="", help='name for global model path')
        
        # For logging
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files and model file, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        
        # For training
        parser.add_argument('--evaluation', action='store_true', default=False, help='evaluation only')
        parser.add_argument('--generate_feature', action='store_true', default=False, help='generate features for training next stage')
        parser.add_argument('--feature_out', type=str, default='features', help='feature output directory')
        parser.add_argument('--restore_features', type=str, default='None', help='feature output directory')
        parser.add_argument('--generate_image', action='store_true', default=False, help='Generate images during evaluation')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--sub_batch_size', type=int, default=6, help='batch size for using local image patches')
        parser.add_argument('--patch_sizes', nargs='+', type=int, help='sizes of patch cropped from big images')
        parser.add_argument('--size', type=int, default=508, help='size (in pixel) for downsampled images')
        parser.add_argument('--origin_size', type=int, default=5000, help='size (in pixel) of the whole images')
        
        parser.add_argument('--training_level', type=int, help='training level (-1 is global only)')
        parser.add_argument('--level_decay', type=float, default=0, help='learning rate decayed through branches')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--reduce_factor', type=float, default=0.2, help='learning rate decay factor')
        parser.add_argument('--reduce_step_size', type=int, default=50, help=' rate decay factor')
        parser.add_argument('--lamb_fmreg', type=float, default=0.15, help='loss weight feature map regularization')
        parser.add_argument('--add_weight',action='store_true', default=False, help='evaluation only')
        parser.add_argument('--supervision',action='store_true', default=False, help='supervision on local branch')
        parser.add_argument('--reg_margin', type=float, default=0.05, help='margin of regularization')
        parser.add_argument('--early_stopping', type=int, default=100, help='number of epochs for early stopping')
        parser.add_argument('--num_epochs', type=int, default=1000, help='number of maximum training epochs')
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
        return args
