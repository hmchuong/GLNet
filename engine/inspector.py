#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils.metrics import ConfusionMatrix

transformer = transforms.Compose([
    transforms.ToTensor(),
])

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    weight = torch.stack([ b['weight'] for b in batch], dim=0)
    
    return {'image': image, 'label': label, 'id': id, 'weight': weight}

def resize(images, shape, label=False):
    """ Resize PIL images
    Parameters
    ----------
    images: list of PIL Images
    shape: tuple (w, h)
    label: resize label or not
    
    Returns
    -------
    list of PIL Images after resizing
    """
    resize_fn = partial(TF.resize, size=shape, interpolation=Image.NEAREST if label else Image.BILINEAR)
    return list(map(resize_fn, images))

def masks_transform(masks, rgb2class, numpy=False):
    """ Transform masks
    
    Parameters
    ----------
    masks: list of PIL images
    rgb2class: function
        convert image numpy array to label array
    numpy: boolean
        convert to numpy or tensor
    
    Returns
    -------
    numpy or tensor
    """
    targets = np.array([rgb2class(np.array(m)).astype('int32') for m in masks], dtype=np.int32)
    if numpy:
        return targets
    return torch.from_numpy(targets).long()

def images_transform(images):
    """ Transform images to tensor
    
    Parameters
    ----------
    images: list of PIL images
    
    Returns
    -------
    torch tensor
    """
    
    inputs = [transformer(img) for img in images]
    inputs = torch.stack(inputs, dim=0)
    return inputs

def get_patch_info(shape, p_size):
    """
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    """
    x = shape[0]
    y = shape[1]
    n = m = 1
    step_size = p_size // 10
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < step_size:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < step_size:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

def slice(images,  p_size):
    """ Slice images, labels and previous output to small patches
    
    Parameters
    ----------
    images: PIL Images or torch.Tensor
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    """
    patches = []
    
    coordinates = []
    templates = []
    sizes = []
    ratios = [(0, 0)] * len(images)
    patch_ones = np.ones(p_size)
    
    for i in range(len(images)):
        if isinstance(images, torch.Tensor):
            h, w = images.shape[2:]
        else:
            w, h = images[i].size
        size = (h, w)
        sizes.append(size)
        ratios[i] = (float(p_size[0]) / size[0], float(p_size[1]) / size[1])
        template = np.zeros(size)
        n_x, n_y, step_x, step_y = get_patch_info(size, p_size[0])
        patches.append([images[i]] * (n_x * n_y))
        coordinates.append([(0, 0)] * (n_x * n_y))
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                template[top:top+p_size[0], left:left+p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                if isinstance(images, torch.Tensor):
                    patches[i][x * n_y + y] = images[i][:, top: top + p_size[0], left: left + p_size[1]]
                else:
                    patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)))
    return patches, coordinates, templates, sizes, ratios

def patches2global(patches, out_size, patch_size, coordinates, templates):
    temp_out = torch.zeros(out_size)
    patches = F.interpolate(patches, size=patch_size, mode='bilinear')
    for coord, patch in zip(coordinates, patches):
        top = int(coord[0]) * out_size[0]
        left = int(coord[1]) * out_size[1]
        temp_out[:, top: top + patch_size[0], left: left + patch_size[1]] += patch
    temp_out /= templates.squeeze(0)
    return temp_out

class Trainer(object):
    
    def __init__(self, device, optimizer, criterion, reg_loss_fn, lamb_reg, training_level, rescale_size, origin_size, patch_sizes, sub_batch_size, rgb2class):
        super(Trainer, self).__init__()
        
        self.device = device
        
        self.optimizer = optimizer
        self.rgb2class = rgb2class
        self.criterion = criterion
        self.reg_loss_fn = reg_loss_fn
        self.lamb_reg = lamb_reg
        self.training_level = training_level
        self.rescale_size = rescale_size
        self.origin_size = origin_size
        self.patch_sizes = patch_sizes
        self.sub_batch_size = sub_batch_size
    
    def train_sub_batches(self, model, patches, label_patches, out_patches, weight_patches, sub_backward, retain_graph, level):
        
        loss = None
        
        # Refine by the local branch
        patches_var = images_transform(patches).to(self.device)
        
        label_patches_var = resize(label_patches, self.rescale_size, label=True)
        label_patches_var = masks_transform(label_patches_var, self.rgb2class)

        out_patches_var = out_patches
        out_patches_var = F.interpolate(out_patches_var, size=self.rescale_size, mode='bilinear')
        
        weight_patches_var = F.interpolate(weight_patches, size=self.rescale_size, mode='bilinear').squeeze()
        
        params = {
            "mode": "local",
            "patches": patches_var.to(self.device),
            "previous_prediction": out_patches_var.to(self.device),
            "level": level
        }
        
        patch_predictions = model(**params)
        
        # Calculate loss if current training level
        if sub_backward:
            self.optimizer.zero_grad()
            loss = (1 - self.lamb_reg) * self.criterion(patch_predictions, label_patches_var.to(self.device), weight_patches_var.to(self.device)) \
                + self.lamb_reg * self.reg_loss_fn(patch_predictions, out_patches_var.to(self.device))
            loss.backward(retain_graph=retain_graph)
            self.optimizer.step()

        # Update back to output_patches
        patch_predictions.require_grad = False
        patch_predictions = patch_predictions.detach()
        return patch_predictions, loss
        
    def train_one_level(self, model, patch_size, images, labels, weights, out, sub_backward, training_level):
            
        # Create local patches
        patches, coordinates, templates, _, ratios = slice(images, patch_size)
        patches = [resize(p_list, self.rescale_size) for p_list in patches]
        
        # Create local labels
        label_patches, _, _, _, _ = slice(labels, patch_size)
        
        # Create local previous prediction
        out_patches, _, _, _, _ = slice(out, patch_size)
        out_patches = [F.interpolate(torch.stack(out_list, dim=0), size=self.rescale_size, mode='bilinear') for out_list in out_patches]
        
        # Create local weights
        weight_patches, _, _, _, _ = slice(weights.unsqueeze(1), patch_size)
        weight_patches = [F.interpolate(torch.stack(out_list, dim=0), size=self.rescale_size, mode='bilinear') for out_list in weight_patches]
        

        for i in range(len(images)):
            j = 0
            while j < len(coordinates[i]):
                patch_predictions, loss = self.train_sub_batches(model, \
                                                                patches[i][j : j+self.sub_batch_size], \
                                                                label_patches[i][j : j+self.sub_batch_size], \
                                                                out_patches[i][j : j+self.sub_batch_size], \
                                                                weight_patches[i][j : j+self.sub_batch_size], \
                                                                sub_backward, \
                                                                not(i == len(images) - 1 and j + self.sub_batch_size >= len(coordinates[i])), \
                                                                training_level)
                out_patches[i][j : j+self.sub_batch_size] = patch_predictions.to("cpu")
                j += self.sub_batch_size
                
            # Update output_patches back to out
            out[i] = patches2global(out_patches[i], out.shape[1:], patch_size, coordinates[i], templates[i])
        
        return out, loss
            
    
    def train(self, model, sample):
        
        images, labels, weights = sample['image'], sample['label'], sample['weight']
        
        # FF global images
        images_glb = resize(images, self.rescale_size)
        images_glb = images_transform(images_glb)
        
        labels_glb = resize(labels, self.rescale_size, label=True)
        labels_glb = masks_transform(labels_glb, self.rgb2class)
        
        params = {
            "mode": "global",
            "images": images_glb.to(self.device)
        }
        global_out = model(**params)
        
        # If training global only
        if self.training_level == -1:
            self.optimizer.zero_grad()
            loss = self.criterion(global_out, labels_glb.to(self.device))
            self.optimizer.step()
            return loss
        
        # Refine result with local branches
        out = F.interpolate(global_out, size=self.origin_size, mode='bilinear').to("cpu")
        
        # For each local branch
        for level in range(self.training_level + 1):
            out, loss = self.train_one_level(model, (self.patch_sizes[level], self.patch_sizes[level]), images, labels, weights, out, (level == self.training_level), level)

        # Return loss of last layer
        return loss         

class Evaluator(object):
    
    def __init__(self, device, sub_batch_size, eval_level, num_classes, patch_sizes, rescaled_size, origin_size, rgb2class):
        super(Evaluator, self).__init__()
        
        self.device = device
        self.eval_level = eval_level
        self.metric = ConfusionMatrix(num_classes)
        self.rescale_size = rescaled_size
        self.origin_size = origin_size
        self.patch_sizes = patch_sizes
        self.rgb2class = rgb2class
        self.sub_batch_size = sub_batch_size
    
    def infer_sub_batches(self, model, patches, out_patches, level):
        
        # Refine by the local branch
        patches_var = images_transform(patches).to(self.device)
        
        out_patches_var = out_patches
        out_patches_var = F.interpolate(out_patches_var, size=self.rescale_size, mode='bilinear')
    
        params = {
            "mode": "local",
            "patches": patches_var.to(self.device),
            "previous_prediction": out_patches_var.to(self.device),
            "level": level
        }
        patch_predictions = model(**params)

        return patch_predictions
        
    def infer_one_level(self, model, patch_size, images, out, training_level):
            
        # Create local patches
        patches, coordinates, templates, _, ratios = slice(images, patch_size)
        patches = [resize(p_list, self.rescale_size) for p_list in patches]
        
        # Create local previous prediction
        out_patches, _, _, _, _ = slice(out, patch_size)
        out_patches = [F.interpolate(torch.stack(out_list, dim=0), size=self.rescale_size, mode='bilinear') for out_list in out_patches]
        
        for i in range(len(images)):
            j = 0
            while j < len(coordinates[i]):
                patch_predictions = self.infer_sub_batches(model, \
                                                                patches[i][j : j+self.sub_batch_size], \
                                                                out_patches[i][j : j+self.sub_batch_size], \
                                                                training_level)
                out_patches[i][j : j+self.sub_batch_size] = patch_predictions.to("cpu")
                j += self.sub_batch_size
                
            # Update output_patches back to out
            out[i] = patches2global(out_patches[i], out.shape[1:], patch_size, coordinates[i], templates[i])
        
        return out
    
    def get_scores(self):
        self.metric.synchronize_between_processes()
        return self.metric.get_scores()

    def reset_metrics(self):
        self.metric.reset()
    
    def eval(self, sample, model):
        model_no_ddp = model
        if hasattr(model, "module") :
            model_no_ddp = model.module
        
        with torch.no_grad():
            images = sample['image']
            
            # FF global images
            images_glb = resize(images, self.rescale_size)
            images_glb = images_transform(images_glb)
            
            params = {
                "mode": "global",
                "images": images_glb.to(self.device)
            }
            global_out = model(**params)
            out = F.interpolate(global_out, size=self.origin_size, mode='bilinear').to("cpu")
            
            # For each local branch
            for level in range(self.eval_level + 1):
                out = self.infer_one_level(model, (self.patch_sizes[level], self.patch_sizes[level]), images, out, level)
            
            out = torch.softmax(out, dim=1).argmax(1).numpy()
            labels = sample['label'] # PIL images
            labels_npy = masks_transform(labels, self.rgb2class, numpy=True)
            self.metric.update(labels_npy, out)
            
            return out
            
        