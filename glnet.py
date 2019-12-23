#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import time

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter

from dataset import *

from utils.loss import CrossEntropyLoss2d, SoftCrossEntropyLoss2d, FocalLoss
from utils.lovasz_losses import lovasz_softmax
from utils.lr_scheduler import LR_Scheduler
from utils.distributed import init_distributed_mode, MetricLogger, reduce_dict, SmoothedValue, save_on_master, is_main_process

from engine import create_model_load_weights, get_optimizer, Trainer, Evaluator, collate, collate_test
from option import Options

def prepare_dataset_loaders(dataset_name, data_path, batch_size, distributed):

    print("preparing datasets and dataloaders......")
    Dataset = eval(dataset_name)
    train_ids, val_ids, test_ids = Dataset.prepare_subset_ids(data_path)

    dataset_train = Dataset(data_path, train_ids, label=True, transform=True)
    dataset_val = Dataset(data_path, val_ids, label=True)
    dataset_test = Dataset(data_path, test_ids, label=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, batch_size, drop_last=True)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=collate)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, sampler=val_sampler, num_workers=args.workers, collate_fn=collate)
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=collate)
    
    return train_sampler, dataloader_train, dataloader_val, dataloader_test

def save_prediction(ids, prediction_dir, dataset, images, labels, predictions_global, predictions_local, predictions):
    for i in range(len(images)):
        img = np.array(images[i])[:,:,[2,1,0]]
        h = img.shape[0]
        line = np.zeros((h, 20, 3))
        line[:, :, 2] = 255
        
        label = np.array(labels[i])[:,:,[2,1,0]]
        img = np.concatenate((img, line), axis=1)
        img = np.concatenate((img, label), axis=1)
        if predictions_global is not None:
            pred = transforms.functional.to_pil_image(dataset.classToRGB(predictions_global[i]) * 255.)
            pred = np.array(pred)[:,:,[2,1,0]]
            img = np.concatenate((img, line), axis=1)
            img = np.concatenate((img, pred), axis=1)
        if predictions_local is not None:
            pred = transforms.functional.to_pil_image(dataset.classToRGB(predictions_local[i]) * 255.)
            pred = np.array(pred)[:,:,[2,1,0]]
            img = np.concatenate((img, line), axis=1)
            img = np.concatenate((img, pred), axis=1)
        if predictions is not None:
            pred = transforms.functional.to_pil_image(dataset.classToRGB(predictions[i]) * 255.)
            pred = np.array(pred)[:,:,[2,1,0]]
            img = np.concatenate((img, line), axis=1)
            img = np.concatenate((img, pred), axis=1)

        cv2.imwrite(os.path.join(prediction_dir, ids[i] + "_result.png"), img)

def main(args):

    # Get properties from argument parser
    n_class = args.n_class
    device = torch.device(args.device)
    data_path = args.data_path
    batch_size = args.batch_size
    
    model_path = args.model_path
    log_path = args.log_path

    # Check and create log directories
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    task_name = args.task_name
    
    mode = args.mode # 1: train global; 2: train local from global; 3: train global from local
    evaluation = args.evaluation
    test = args.test

    print("mode:", mode, "evaluation:", evaluation, "test:", test)

    ##### sizes are (w, h) ##############################
    # make sure margin / 32 is over 1.5 AND size_g is divisible by 4
    size_g = (args.size_g, args.size_g) # resized global image
    size_p = (args.size_p, args.size_p) # cropped local patch size
    sub_batch_size = args.sub_batch_size # batch size for train local patches

    # Prepare data loaders
    train_sampler, dataloader_train, dataloader_val, dataloader_test = prepare_dataset_loaders(args.dataset, data_path, batch_size, args.distributed)

    # Create models
    print("creating models......")

    path_g = os.path.join(model_path, args.path_g)
    path_g2l = os.path.join(model_path, args.path_g2l)
    path_l2g = os.path.join(model_path, args.path_l2g)
    model, global_fixed = create_model_load_weights(n_class, args.distributed, device, args.gpu, mode, evaluation, path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g)

    # Training config
    ###################################
    num_epochs = args.num_epochs
    learning_rate = args.lr
    lamb_fmreg = args.lamb_fmreg

    optimizer = get_optimizer(model, mode, learning_rate=learning_rate)

    scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))
    ##################################

    # Loss functions
    criterion1 = FocalLoss(gamma=3)
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = lovasz_softmax
    criterion = lambda x,y: criterion1(x, y)
    mse = nn.MSELoss()

    # Logging when training
    if not evaluation and is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(log_path, task_name))
        f_log = open(os.path.join(log_path, task_name + ".log"), 'w')

    trainer = Trainer(device, criterion, optimizer, n_class, size_g, size_p, sub_batch_size, mode, lamb_fmreg)
    evaluator = Evaluator(device, n_class, size_g, size_p, sub_batch_size, mode, test)

    best_pred = 0.0
    print("start training......")
    for epoch in range(num_epochs):

        # Train one epoch
        #########################################################
        if args.distributed:
            train_sampler.set_epoch(epoch)
        trainer.set_train(model)
        optimizer.zero_grad()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        for i_batch, sample_batched in enumerate(metric_logger.log_every(dataloader_train, 1, header)):
            if evaluation: break
            scheduler(optimizer, i_batch, epoch, best_pred)
            loss = trainer.train(sample_batched, model, global_fixed, dataloader_train.dataset.RGBToClass)
            metric_logger.update(loss=loss)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #########################################################

        # Evaluate after training
        #########################################################
        with torch.no_grad():
            
            print("evaluating...")
            
            n_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            
            model.eval()
            
            metric_logger = MetricLogger(delimiter="  ")
            header = 'Test:'
            
            if test: data_loader = dataloader_test
            else: data_loader = dataloader_val
            
            for i_batch, sample_batched in enumerate(metric_logger.log_every(data_loader, 1, header)):
                torch.cuda.synchronize()
                model_time = time.time()
                predictions, predictions_global, predictions_local = evaluator.eval_test(sample_batched, model, global_fixed, data_loader.dataset.RGBToClass)
                model_time = time.time() - model_time
                metric_logger.update(model_time=model_time)
                
                images = sample_batched['image']
                labels = sample_batched['label']

                # Save the result if testing mode
                if test:
                    prediction_dir = f"./prediction/{task_name}"
                    os.makedirs(prediction_dir, exist_ok=True)
                    save_prediction(sample_batched['id'], prediction_dir, data_loader.dataset, images, labels, predictions_global, predictions_local, predictions)

                # Log the image to the tensorboard  
                if not evaluation and not test and is_main_process():
                    if i_batch * batch_size + len(images) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (epoch % len(dataloader_val)):
                        writer.add_image('image', transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)
                        if not test:
                            writer.add_image('mask', data_loader.dataset.classToRGB(np.array(labels[(epoch % len(dataloader_val)) - i_batch * batch_size])) * 255., epoch)
                        if mode == 2 or mode == 3:
                            writer.add_image('prediction', data_loader.dataset.classToRGB(predictions[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                            writer.add_image('prediction_local', data_loader.dataset.classToRGB(predictions_local[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                        writer.add_image('prediction_global', data_loader.dataset.classToRGB(predictions_global[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)            
            
            #########################################################

            if test: break
            
            score_val, score_val_global, score_val_local = evaluator.get_scores()
            evaluator.reset_metrics()
            current_score = score_val_global["iou"][1:] if mode == 1 else score_val["iou"][1:]
            current_score = np.mean(np.nan_to_num(current_score))
            if current_score > best_pred: 
                best_pred = current_score
                if not (test or evaluation): 
                    save_on_master(model.module.state_dict(), os.path.join(args.model_path, task_name + ".pth"))

            log = ""
            log = log + 'epoch [{}/{}] IoU: {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_val["iou"][1:]))) + "\n"
            log = log + 'epoch [{}/{}] Local  -- IoU: val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_val_local["iou"][1:]))) + "\n"
            log = log + 'epoch [{}/{}] Global -- IoU: val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_val_global["iou"][1:]))) + "\n"
            log = log + "val:" + str(score_val["iou"]) + "\n"
            log = log + "Local val:" + str(score_val_local["iou"]) + "\n"
            log = log + "Global val:" + str(score_val_global["iou"]) + "\n"
            log += "================================\n"
            print(log)

            if evaluation: break

            f_log.write(log)
            f_log.flush()
            if is_main_process():
                writer.add_scalars('IoU', {'validation iou': current_score}, epoch)
            torch.set_num_threads(n_threads)

    if not evaluation and is_main_process(): f_log.close()

if __name__ == "__main__":
    args = Options().parse()
    init_distributed_mode(args)
    main(args)