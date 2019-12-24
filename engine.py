#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

from functools import partial

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.distributed as dist

from models.fpn_global_local_fmreg_ensemble import fpn
from utils.metrics import ConfusionMatrix

transformer = transforms.Compose([
    transforms.ToTensor(),
])

def emap(fn, iterable):
    """eager map because I'm lazy and don't want to type."""
    return list(map(fn, iterable))

def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resize_fn = partial(TF.resize, size=shape, interpolation=Image.NEAREST if label else Image.BILINEAR)
    return emap(resize_fn, images)

def masks_transform(masks, device, rgb2class, numpy=False):
    '''
    masks: list of PIL images
    '''
    targets = np.array([rgb2class(np.array(m)).astype('int32') for m in masks], dtype=np.int32)
    if numpy:
        return targets
    return torch.from_numpy(targets).long().to(device)

def images_transform(images, device):
    '''
    images: list of PIL images
    '''
    
    inputs = [transformer(img) for img in images]
    inputs = torch.stack(inputs, dim=0).to(device)
    return inputs

def get_patch_info(shape, p_size):
    '''
    shape: origin image size, (x, y)
    p_size: patch size (square)
    return: n_x, n_y, step_x, step_y
    '''
    x = shape[0]
    y = shape[1]
    n = m = 1
    while x > n * p_size:
        n += 1
    while p_size - 1.0 * (x - p_size) / (n - 1) < 50:
        n += 1
    while y > m * p_size:
        m += 1
    while p_size - 1.0 * (y - p_size) / (m - 1) < 50:
        m += 1
    return n, m, (x - p_size) * 1.0 / (n - 1), (y - p_size) * 1.0 / (m - 1)

def global2patch(images, p_size, device):
    '''
    image/label => patches
    p_size: patch size
    return: list of PIL patch images; coordinates: images->patches; ratios: (h, w)
    '''
    patches = []; coordinates = []; templates = []; sizes = []; ratios = [(0, 0)] * len(images); patch_ones = np.ones(p_size)
    for i in range(len(images)):
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
                patches[i][x * n_y + y] = transforms.functional.crop(images[i], top, left, p_size[0], p_size[1])
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)).to(device))
    return patches, coordinates, templates, sizes, ratios

def patch2global(patches, n_class, sizes, coordinates, p_size):
    '''
    predicted patches (after classify layer) => predictions
    return: list of np.array
    '''
    predictions = [ np.zeros((n_class, size[0], size[1])) for size in sizes ]
    for i in range(len(sizes)):
        for j in range(len(coordinates[i])):
            top, left = coordinates[i][j]
            top = int(np.round(top * sizes[i][0])); left = int(np.round(left * sizes[i][1]))
            predictions[i][:, top: top + p_size[0], left: left + p_size[1]] += patches[i][j]
    return predictions

def template_patch2global(size_g, size_p, n, step, device):
    template = np.zeros(size_g)
    coordinates = [(0, 0)] * n ** 2
    patch = np.ones(size_p)
    step = (size_g[0] - size_p[0]) // (n - 1)
    x = y = 0
    i = 0
    while x + size_p[0] <= size_g[0]:
        while y + size_p[1] <= size_g[1]:
            template[x:x+size_p[0], y:y+size_p[1]] += patch
            coordinates[i] = (1.0 * x / size_g[0], 1.0 * y / size_g[1])
            i += 1
            y += step
        x += step
        y = 0
    return Variable(torch.Tensor(template).expand(1, 1, -1, -1)).to(device), coordinates

def one_hot_gaussian_blur(index, classes):
    '''
    index: numpy array b, h, w
    classes: int
    '''
    mask = np.transpose((np.arange(classes) == index[..., None]).astype(float), (0, 3, 1, 2))
    b, c, _, _ = mask.shape
    for i in range(b):
        for j in range(c):
            mask[i][j] = cv2.GaussianBlur(mask[i][j], (0, 0), 8)

    return mask

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    return {'image': image, 'label': label, 'id': id}

def create_model_load_weights(n_class, distributed, device, gpu, mode=1, evaluation=False, path_g=None, path_g2l=None, path_l2g=None):
    model = fpn(n_class)
    model = model.to(device)
    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if (mode == 2 and not evaluation) or (mode == 1 and evaluation):
        # load fixed basic global branch
        partial = torch.load(path_g, map_location='cpu')
        state = model_without_ddp.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model_without_ddp.load_state_dict(state)

    if (mode == 3 and not evaluation) or (mode == 2 and evaluation):
        partial = torch.load(path_g2l,)
        state = model_without_ddp.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model_without_ddp.load_state_dict(state)

    global_fixed = None
    if mode == 3:
        # load fixed basic global branch
        global_fixed = fpn(n_class)
        global_fixed = global_fixed.to(device)
        glob_without_ddp = global_fixed
        if distributed:
            global_fixed = torch.nn.parallel.DistributedDataParallel(global_fixed, device_ids=[gpu], find_unused_parameters=True)
            glob_without_ddp = global_fixed.module
        partial = torch.load(path_g, map_location='cpu')
        state = glob_without_ddp.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state and "local" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        glob_without_ddp.load_state_dict(state)
        global_fixed.eval()

    if mode == 3 and evaluation:
        partial = torch.load(path_l2g, map_location='cpu')
        state = model_without_ddp.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in partial.items() if k in state}# and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model_without_ddp.load_state_dict(state)

    if mode == 1 or mode == 3:
        model_without_ddp.resnet_local.eval()
        model_without_ddp.fpn_local.eval()
    else:
        model_without_ddp.resnet_global.eval()
        model_without_ddp.fpn_global.eval()
    
    return model, global_fixed


def get_optimizer(model, mode=1, learning_rate=2e-5):
    model_no_ddp = model
    if hasattr(model, "module") :
        model_no_ddp = model.module
    if mode == 1 or mode == 3:
        # train global
        
        optimizer = torch.optim.Adam([
                {'params': model_no_ddp.resnet_global.parameters(), 'lr': learning_rate},
                {'params': model_no_ddp.resnet_local.parameters(), 'lr': 0},
                {'params': model_no_ddp.fpn_global.parameters(), 'lr': learning_rate},
                {'params': model_no_ddp.fpn_local.parameters(), 'lr': 0},
                {'params': model_no_ddp.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)

    else:
        # train local
        optimizer = torch.optim.Adam([
                {'params': model_no_ddp.resnet_global.parameters(), 'lr': 0},
                {'params': model_no_ddp.resnet_local.parameters(), 'lr': learning_rate},
                {'params': model_no_ddp.fpn_global.parameters(), 'lr': 0},
                {'params': model_no_ddp.fpn_local.parameters(), 'lr': learning_rate},
                {'params': model_no_ddp.ensemble_conv.parameters(), 'lr': learning_rate},
            ], weight_decay=5e-4)
    return optimizer


class Trainer(object):
    def __init__(self, device, criterion, optimizer, n_class, size_g, size_p, sub_batch_size=6, mode=1, lamb_fmreg=0.15):
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_class = n_class
        self.size_g = size_g
        self.size_p = size_p
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.lamb_fmreg = lamb_fmreg
        self.device = device
    
    def set_train(self, model):
        model_no_ddp = model
        if hasattr(model, "module") :
            model_no_ddp = model.module
        model_no_ddp.ensemble_conv.train()
        if self.mode == 1 or self.mode == 3:
            model_no_ddp.resnet_global.train()
            model_no_ddp.fpn_global.train()
        else:
            model_no_ddp.resnet_local.train()
            model_no_ddp.fpn_local.train()

    def train(self, sample, model, global_fixed, rgb2class):
        model_no_ddp = model
        if hasattr(model, "module") :
            model_no_ddp = model.module

        images, labels = sample['image'], sample['label'] # PIL images
        images_glb = resize(images, self.size_g) # list of resized PIL images
        images_glb = images_transform(images_glb, self.device)
        labels_glb = resize(labels, (self.size_g[0] // 4, self.size_g[1] // 4), label=True) # FPN down 1/4, for loss
        labels_glb = masks_transform(labels_glb, self.device, rgb2class)

        if self.mode == 2 or self.mode == 3:
            patches, coordinates, templates, _, ratios = global2patch(images, self.size_p, self.device)
            label_patches, _, _, _, _ = global2patch(labels, self.size_p, self.device)

        if self.mode == 1:
            # training with only (resized) global image #########################################
            outputs_global, _ = model.forward(images_glb, None, None, None)
            #print(outputs_global.shape, labels_glb.shape)
            #print(outputs_global, labels_glb)
            #import pdb; pdb.set_trace()
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            ##############################################

        if self.mode == 2:
            # training with patches ###########################################
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size], self.device) # b, c, h, w
                    label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], (self.size_p[0] // 4, self.size_p[1] // 4), label=True), self.device, rgb2class) # down 1/4 for loss

                    output_ensembles, _, output_patches, fmreg_l2 = model.forward(images_glb[i:i+1], patches_var, coordinates[i][j : j+self.sub_batch_size], ratios[i], mode=self.mode, n_patch=len(coordinates[i]))
                    loss = self.criterion(output_patches, label_patches_var) + self.criterion(output_ensembles, label_patches_var) + self.lamb_fmreg * fmreg_l2
                    loss.backward()
                    j += self.sub_batch_size

            self.optimizer.step()
            self.optimizer.zero_grad()
            #####################################################################################

        if self.mode == 3:
            # train global with help from patches ##################################################
            # go through local patches to collect feature maps
            # collect predictions from patches
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    patches_var = images_transform(patches[i][j : j+self.sub_batch_size], self.device) # b, c, h, w
                    fm_patches, _ = model_no_ddp.collect_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, template=templates[i], n_patch_all=len(coordinates[i]))
                    j += self.sub_batch_size

            # train on global image
            outputs_global, fm_global = model.forward(images_glb, None, None, None, mode=self.mode)
            loss = self.criterion(outputs_global, labels_glb)
            loss.backward(retain_graph=True)
            # fmreg loss
            # generate ensembles & calc loss
            for i in range(len(images)):
                j = 0
                while j < len(coordinates[i]):
                    label_patches_var = masks_transform(resize(label_patches[i][j : j+self.sub_batch_size], (self.size_p[0] // 4, self.size_p[1] // 4), label=True), self.device, rgb2class)
                    fl = fm_patches[i][j : j+self.sub_batch_size].to(self.device)
                    fg = model.module._crop_global(fm_global[i:i+1], coordinates[i][j:j+self.sub_batch_size], ratios[i])[0]
                    fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                    output_ensembles = model.module.ensemble(fl, fg)
                    loss = self.criterion(output_ensembles, label_patches_var)# + 0.15 * mse(fl, fg)
                    if i == len(images) - 1 and j + self.sub_batch_size >= len(coordinates[i]):
                        loss.backward()
                    else:
                        loss.backward(retain_graph=True)

                    # ensemble predictions
                    j += self.sub_batch_size
            self.optimizer.step()
            self.optimizer.zero_grad()
        dist.all_reduce(loss)
        return loss


class Evaluator(object):
    def __init__(self, device, n_class, size_g, size_p, sub_batch_size=6, mode=1, test=False):
        self.metrics_global = ConfusionMatrix(n_class)
        self.metrics_local = ConfusionMatrix(n_class)
        self.metrics = ConfusionMatrix(n_class)
        self.n_class = n_class
        self.size_g = size_g
        self.size_p = size_p
        self.sub_batch_size = sub_batch_size
        self.mode = mode
        self.test = test
        self.device = device

        if test:
            self.flip_range = [False, True]
            self.rotate_range = [0, 1, 2, 3]
        else:
            self.flip_range = [False]
            self.rotate_range = [0]
    
    def get_scores(self):
        self.metrics.synchronize_between_processes()
        score_train = self.metrics.get_scores()
        self.metrics_local.synchronize_between_processes()
        score_train_local = self.metrics_local.get_scores()
        self.metrics_global.synchronize_between_processes()
        score_train_global = self.metrics_global.get_scores()
        return score_train, score_train_global, score_train_local

    def reset_metrics(self):
        self.metrics.reset()
        self.metrics_local.reset()
        self.metrics_global.reset()

    def eval_test(self, sample, model, global_fixed, rgb2class):
        model_no_ddp = model
        if hasattr(model, "module") :
            model_no_ddp = model.module
        with torch.no_grad():
            images = sample['image']
            if not self.test:
                labels = sample['label'] # PIL images
                labels_npy = masks_transform(labels, self.device, rgb2class, numpy=True)

            images_global = resize(images, self.size_g)
            outputs_global = np.zeros((len(images), self.n_class, self.size_g[0] // 4, self.size_g[1] // 4))
            if self.mode == 2 or self.mode == 3:
                images_local = [ image.copy() for image in images ]
                scores_local = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]
                scores = [ np.zeros((1, self.n_class, images[i].size[1], images[i].size[0])) for i in range(len(images)) ]

            for flip in self.flip_range:
                if flip:
                    # we already rotated images for 270'
                    for b in range(len(images)):
                        images_global[b] = transforms.functional.rotate(images_global[b], 90) # rotate back!
                        images_global[b] = transforms.functional.hflip(images_global[b])
                        if self.mode == 2 or self.mode == 3:
                            images_local[b] = transforms.functional.rotate(images_local[b], 90) # rotate back!
                            images_local[b] = transforms.functional.hflip(images_local[b])
                for angle in self.rotate_range:
                    if angle > 0:
                        for b in range(len(images)):
                            images_global[b] = transforms.functional.rotate(images_global[b], 90)
                            if self.mode == 2 or self.mode == 3:
                                images_local[b] = transforms.functional.rotate(images_local[b], 90)

                    # prepare global images onto cuda
                    images_glb = images_transform(images_global, self.device) # b, c, h, w

                    if self.mode == 2 or self.mode == 3:
                        patches, coordinates, templates, sizes, ratios = global2patch(images, self.size_p, self.device)
                        predicted_patches = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]
                        predicted_ensembles = [ np.zeros((len(coordinates[i]), self.n_class, self.size_p[0], self.size_p[1])) for i in range(len(images)) ]

                    if self.mode == 1:
                        # eval with only resized global image ##########################
                        if flip:
                            outputs_global += np.flip(np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global += np.rot90(model.forward(images_glb, None, None, None)[0].data.cpu().numpy(), k=angle, axes=(3, 2))
                        ################################################################

                    if self.mode == 2:
                        # eval with patches ###########################################
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size], self.device) # b, c, h, w
                                output_ensembles, output_global, output_patches, _ = model.forward(images_glb[i:i+1], patches_var, coordinates[i][j : j+self.sub_batch_size], ratios[i], mode=self.mode, n_patch=len(coordinates[i]))

                                # patch predictions
                                predicted_patches[i][j:j+output_patches.size()[0]] += F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                                predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += patches_var.size()[0]
                            if flip:
                                outputs_global[i] += np.flip(np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1)), axis=2)
                                scores_local[i] += np.flip(np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                                scores[i] += np.flip(np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3) # merge softmax scores from patches (overlaps)
                            else:
                                outputs_global[i] += np.rot90(output_global[0].data.cpu().numpy(), k=angle, axes=(2, 1))
                                scores_local[i] += np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores[i] += np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                        ###############################################################

                    if self.mode == 3:
                        # eval global with help from patches ##################################################
                        # go through local patches to collect feature maps
                        # collect predictions from patches
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                patches_var = images_transform(patches[i][j : j+self.sub_batch_size], self.device) # b, c, h, w
                                fm_patches, output_patches = model_no_ddp.collect_local_fm(images_glb[i:i+1], patches_var, ratios[i], coordinates[i], [j, j+self.sub_batch_size], len(images), global_model=global_fixed, template=templates[i], n_patch_all=len(coordinates[i]))
                                predicted_patches[i][j:j+output_patches.size()[0]] += F.interpolate(output_patches, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += self.sub_batch_size
                        # go through global image
                        tmp, fm_global = model.forward(images_glb, None, None, None, mode=self.mode)
                        if flip:
                            outputs_global += np.flip(np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2)), axis=3)
                        else:
                            outputs_global += np.rot90(tmp.data.cpu().numpy(), k=angle, axes=(3, 2))
                        # generate ensembles
                        for i in range(len(images)):
                            j = 0
                            while j < len(coordinates[i]):
                                fl = fm_patches[i][j : j+self.sub_batch_size].cuda()
                                fg = model_no_ddp._crop_global(fm_global[i:i+1], coordinates[i][j:j+self.sub_batch_size], ratios[i])[0]
                                fg = F.interpolate(fg, size=fl.size()[2:], mode='bilinear')
                                output_ensembles = model_no_ddp.ensemble(fl, fg) # include cordinates

                                # ensemble predictions
                                predicted_ensembles[i][j:j+output_ensembles.size()[0]] += F.interpolate(output_ensembles, size=self.size_p, mode='nearest').data.cpu().numpy()
                                j += self.sub_batch_size
                            if flip:
                                scores_local[i] += np.flip(np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3)[0] # merge softmax scores from patches (overlaps)
                                scores[i] += np.flip(np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)), axis=3)[0] # merge softmax scores from patches (overlaps)
                            else:
                                scores_local[i] += np.rot90(np.array(patch2global(predicted_patches[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                                scores[i] += np.rot90(np.array(patch2global(predicted_ensembles[i:i+1], self.n_class, sizes[i:i+1], coordinates[i:i+1], self.size_p)), k=angle, axes=(3, 2)) # merge softmax scores from patches (overlaps)
                        ###################################################

            # global predictions ###########################
            outputs_global = torch.Tensor(outputs_global)
            predictions_global = [F.interpolate(outputs_global[i:i+1], images[i].size[::-1], mode='nearest').argmax(1).detach().numpy()[0] for i in range(len(images))]
            if not self.test:
                labels = [rgb2class(np.array(label)) for label in sample['label']]
                self.metrics_global.update(labels_npy, predictions_global)

            if self.mode == 2 or self.mode == 3:
                # patch predictions ###########################
                predictions_local = [ score.argmax(1)[0] for score in scores_local ]
                if not self.test:
                    self.metrics_local.update(labels_npy, predictions_local)
                ###################################################
                # combined/ensemble predictions ###########################
                predictions = [ score.argmax(1)[0] for score in scores ]
                if not self.test:
                    self.metrics.update(labels_npy, predictions)
                return predictions, predictions_global, predictions_local
            else:
                return None, predictions_global, None