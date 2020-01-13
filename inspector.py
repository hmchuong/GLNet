#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import time
import json

from tensorboardX import SummaryWriter

import cv2
import numpy as np

import torch
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler, DataLoader

from dataset import *
from engine.inspector import collate, Trainer, Evaluator
from models.inspector_net import InspectorNet
from option.inspector import Options
from utils.distributed import init_distributed_mode, is_main_process, MetricLogger, SmoothedValue, save_on_master
from utils.loss import FocalLoss, MSELossWithMargin

def prepare_dataset_loaders(dataset_name, data_path, batch_size, distributed, workers, **args):
    """ Prepare dataset loaders
    
    Parameters
    ----------
    dataset_name: str
        Dataset name to create
    data_path: str
        path to data
    batch_size: int
        batch size
    distributed: boolean
        distributed training or not
    workers: int
        number of workers for data loading
    
    Returns
    -------
    BatchSampler
        training sampler
    DataLoader
        train dataloader
    DataLoader
        evaluation dataloader
    DataLoader
        test dataloader
    
    """
    print("Preparing datasets and dataloaders......")
    Dataset = eval(dataset_name)
    (train_ids, train_data_path), (val_ids, val_data_path), (test_ids, test_data_path) = Dataset.prepare_subset_ids(data_path)

    # Create datasets
    dataset_train = Dataset(train_data_path, train_ids, label=True, transform=True)
    dataset_val = Dataset(val_data_path, val_ids, label=True)
    dataset_test = Dataset(test_data_path, test_ids, label=True)

    # Create dataset sampler
    if distributed:
        train_sampler = DistributedSampler(dataset_train)
        val_sampler = DistributedSampler(dataset_val)
        test_sampler = DistributedSampler(dataset_test)
    else:
        train_sampler = RandomSampler(dataset_train)
        val_sampler = SequentialSampler(dataset_val)
        test_sampler = SequentialSampler(dataset_test)

    # Create batch sampler for training
    train_batch_sampler = BatchSampler(
                train_sampler, batch_size, drop_last=True)
    
    # Create data loaders
    dataloader_train = DataLoader(dataset=dataset_train, batch_sampler=train_batch_sampler, num_workers=workers, collate_fn=collate)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, sampler=val_sampler, num_workers=workers, collate_fn=collate)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, sampler=test_sampler, num_workers=workers, collate_fn=collate)
    
    return train_sampler, dataloader_train, dataloader_val, dataloader_test

def train(trainer, model, data_loader, optimizer, epoch):
    """ Train the model
    """
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for i_batch, sample_batched in enumerate(metric_logger.log_every(data_loader, 1, header)):
        # Train
        loss = trainer.train(model, sample_batched)
        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

def save_prediction(ids, prediction_dir, classToRGB, images, labels, predictions):
    for i in range(len(images)):
        img = np.array(images[i])[:,:,[2,1,0]]
        h = img.shape[0]
        line = np.zeros((h, 20, 3))
        line[:, :, 2] = 255
        
        label = np.array(labels[i])[:,:,[2,1,0]]
        img = np.concatenate((img, line), axis=1)
        img = np.concatenate((img, label), axis=1)
        if predictions is not None:
            pred = transforms.functional.to_pil_image(classToRGB(predictions[i]) * 255.)
            pred = np.array(pred)[:,:,[2,1,0]]
            img = np.concatenate((img, line), axis=1)
            img = np.concatenate((img, pred), axis=1)

        cv2.imwrite(os.path.join(prediction_dir, ids[i] + "_result.png"), img)

def evaluate(evaluator, model, data_loader, generate_image, writer, epoch, num_epochs, task_name):
    """ Evaluate the model
    """
    print("Evaluating...")
    score = 0
    with torch.no_grad():
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        model.eval()
        
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        
        for i_batch, sample_batched in enumerate(metric_logger.log_every(data_loader, 1, header)):
            torch.cuda.synchronize()
            model_time = time.time()
            # Evaluate and get the prediction
            out = evaluator.eval(sample_batched, model)
            model_time = time.time() - model_time
            metric_logger.update(model_time=model_time)
            
            if generate_image:
                # Save the images
                prediction_dir = f"./prediction/{task_name}"
                os.makedirs(prediction_dir, exist_ok=True)
                images = sample_batched['image']
                labels = sample_batched['label']
                save_prediction(sample_batched['id'], prediction_dir, data_loader.dataset.classToRGB, images, labels, out)
        
        score_val = evaluator.get_scores()
        evaluator.reset_metrics()
        score = score_val["iou"][1:]
        score = np.mean(np.nan_to_num(score))
        
        # Log the results
        log = ""
        log = log + 'epoch [{}/{}] IoU: {:.4f}'.format(epoch+1, num_epochs, score) + "\n"
        log = log + "val:" + str(score_val["iou"]) + "\n"
        log += "================================\n"
        print(log)
        if is_main_process():
            writer.add_scalars('IoU', {'validation iou': score}, epoch)
        torch.set_num_threads(n_threads)
    
    return score

def main(args):
    
    # Create log path
    task_name = args.task_name
    log_path = os.path.join(args.log_path, task_name)
    os.makedirs(log_path, exist_ok=True)
    
    # Check mode
    evaluation = args.evaluation
    generate_image = args.generate_image
    
    # Create dataloader
    train_sampler, dataloader_train, dataloader_val, dataloader_test = prepare_dataset_loaders(**vars(args))
    
    # Create model
    print("Creating models")
    device = args.device
    distributed = args.distributed
    
    model = InspectorNet(args.n_class, args.num_scaling_level, backbone=args.backbone, attention=args.attention)
    model = model.to(device)
    model_without_ddp = model
    gpu = getattr(args, 'gpu', 0)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        
    # Restore model
    if os.path.isfile(args.restore_path):
        print("Restoring...")
        state = torch.load(args.restore_path, map_location='cpu')
        model_without_ddp.load_state_dict(state, strict=False)
        if not evaluation and args.training_level != -1:
            model_without_ddp.copy_weight(args.training_level - 1, args.training_level)
            print("Copy weight from previous training branch...")
    
    # Create logger
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=log_path)
        with open(os.path.join(log_path, "params.json"), "w") as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
    
    # Create evaluator
    training_level = args.training_level
    evaluator = Evaluator(device, args.sub_batch_size, training_level, args.n_class, args.patch_sizes, (args.size, args.size), (args.origin_size, args.origin_size), dataloader_test.dataset.RGBToClass)
    
    if evaluation:
        evaluate(evaluator, model, dataloader_test, generate_image, writer, 0,0, task_name)
        return
    
    # Training
    ###############################################
    
    # Training config
    num_epochs = args.num_epochs
    params = model_without_ddp.get_training_parameters(training_level, decay_rate=args.level_decay, learning_rate=args.lr)
    
    optimizer = torch.optim.Adam(params)
    print("Number of training parameters:", sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad))
    
    criterion = FocalLoss(gamma=3, add_weight=args.add_weight)
    reg_loss = MSELossWithMargin(margin=args.reg_margin)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.reduce_step_size, gamma=args.reduce_factor)
    
    # Create trainer
    trainer = Trainer(device, optimizer, criterion, reg_loss, args.lamb_fmreg, training_level, (args.size, args.size), (args.origin_size, args.origin_size), args.patch_sizes, args.sub_batch_size, dataloader_train.dataset.RGBToClass)
    
    best_score = 0.0
    print("Start training")
    non_improved_epoch = 0
    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        train(trainer, model, dataloader_train, optimizer, epoch)
        
        # Evaluate one epoch
        score = evaluate(evaluator, model, dataloader_val, False, writer, epoch, num_epochs, task_name)
        
        lr_scheduler.step()
        
        # Save model
        if not is_main_process():
            continue
        
        if score > best_score:
            best_score = score
            save_on_master(model_without_ddp.state_dict(), os.path.join(log_path, task_name + ".pth"))
            non_improved_epoch = 0
            continue
        
        non_improved_epoch += 1
        if non_improved_epoch > args.early_stopping:
            break
            
if __name__ == "__main__":
    args = Options().parse()
    init_distributed_mode(args)
    if is_main_process():
        print(args)
    main(args)
    