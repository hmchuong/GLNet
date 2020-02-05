#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

from functools import partial, reduce

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage import transform as sk_transform

from utils.metrics import ConfusionMatrix
from utils.parallel import map_parallel

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
    return map_parallel(lambda image: TF.resize(image, size=shape, interpolation=Image.NEAREST if label else Image.BILINEAR), images)

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
    targets = np.array(map_parallel(lambda m: rgb2class(np.array(m)).astype('int32'), masks), dtype=np.int32)
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
    
    inputs = map_parallel(lambda img: transformer(img), images)
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

def tanh_warping(image, top, left, width, height):
    tanh_one = 0.76159415595
    image_np = None
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1,2,0).to("cpu").numpy()
    else:
        image_np = np.array(image)
    src = np.array([[left, top], [left, top + height], [left + width, top], [left + width, top + height]])
    dst = np.array([[-1,-1], [-1, 1], [1, -1], [1, 1]])
    tform = sk_transform.estimate_transform('similarity', src, dst)
    
    def map_func(coords):
        tform2 = sk_transform.SimilarityTransform(scale=1./257., rotation=0, translation=(-0.99, -0.99))
        return tform.inverse(np.arctanh(tform2(coords)))

    warped = sk_transform.warp(image_np, inverse_map=map_func, output_shape=[height, width] )
    
    if isinstance(image, torch.Tensor):
        return torch.Tensor(warped).permute(2,0,1)
    else:
        return Image.fromarray(warped)

def extract_patch_from_warps(image):
    tanh_one = 0.76159415595
    _, _, h, w = image.shape
    
    top = int(h * (1- tanh_one))
    bottom = h - top
    left = int(w * (1- tanh_one))
    right = w - left
    
    patch = F.interpolate(image[:, :, top: bottom, left: right], size=(w, h), mode='bilinear')
    return patch

def cropping(image, top, left, width, height):
    if isinstance(image, torch.Tensor):
        return image[:, top: top + height, left: left + width]
    else:
        return transforms.functional.crop(image, top, left, width, height)

def patching(image, top, left, width, height, warping=False):
    if warping:
        return tanh_warping(image, top, left, width, height)
    return cropping(image, top, left, width, height)

def slice_only(images, p_size, n_x, n_y, step_x, step_y):
    patches = []
    
    for i in range(len(images)):
        if isinstance(images, torch.Tensor):
            h, w = images.shape[2:]
        else:
            w, h = images[i].size
        size = (h, w)
        patches.append([images[i]] * (n_x * n_y))
        tops = []
        lefts = []
        
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                tops.append(top)
                lefts.append(left)
        patches[i] = list(map(lambda x: patching(images[i], x[0], x[1], p_size[0], p_size[1]), zip(tops, lefts)))
        #patches[i][x * n_y + y] = patching(images[i], top, left, p_size[0], p_size[1])
    return patches

def slice(images,  p_size, tanh_warping=False):
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
        tops = []
        lefts = []
        for x in range(n_x):
            if x < n_x - 1: top = int(np.round(x * step_x))
            else: top = size[0] - p_size[0]
            for y in range(n_y):
                if y < n_y - 1: left = int(np.round(y * step_y))
                else: left = size[1] - p_size[1]
                template[top:top+p_size[0], left:left+p_size[1]] += patch_ones
                coordinates[i][x * n_y + y] = (1.0 * top / size[0], 1.0 * left / size[1])
                #patches[i][x * n_y + y] = patching(images[i], top, left, p_size[0], p_size[1])
                tops.append(top)
                lefts.append(left)
        patches[i] = list(map(lambda x: patching(images[i], x[0], x[1], p_size[0], p_size[1]), zip(tops, lefts)))
        templates.append(Variable(torch.Tensor(template).expand(1, 1, -1, -1)))
    return patches, coordinates, templates, sizes, ratios, n_x, n_y, step_x, step_y

def patches2global(patches, out_size, patch_size, coordinates, templates):
    temp_out = torch.zeros(out_size)
    patches = F.interpolate(patches, size=patch_size, mode='bilinear')
    i = 0
    for coord, patch in zip(coordinates, patches):
        top = int(coord[0] * out_size[1])
        left = int(coord[1] * out_size[2])
        temp_out[:, top: top + patch_size[0], left: left + patch_size[1]] += patch
        i += 1
    temp_out /= templates.squeeze(0)
    return temp_out

def crop_global_features(out_features, coordinates, ratios):
    """
    Parameters
    ----------
    out_features: ((C x H x W) x 7)
    coordinates: [(top, left) x N]
    ratios: [(width, height) x N]
    
    Returns
    -------
    patch_features: [(N x C x H x W) x 7]
    """
    patch_features = []
    for feat in out_features:
        if not isinstance(feat, list):
            feats = [feat]
        else:
            feats = feat
        temp_feats = []
        for feat in feats:
            cropped_feats = []
            for coord in coordinates:
                _, H, W = feat.shape
                top = int(coord[0] * H)
                bottom = top + int(ratios[0] * H)
                left = int(coord[1] * W)
                right = left + int(ratios[1] * W)
                cropped_feats.append(feat[:, top: bottom, left: right])
            cropped_feats = torch.stack(cropped_feats)
            temp_feats.append(cropped_feats)
        patch_features.append(temp_feats[0] if len(temp_feats) == 1 else temp_feats)
    return patch_features

class Trainer(object):
    
    def __init__(self, device, optimizer, criterion, reg_loss_fn, lamb_reg, training_level, rescale_size, origin_size, patch_sizes, sub_batch_size, rgb2class, supervision, warping, glob2local):
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
        self.supervision = supervision
        self.warping = warping
        self.glob2local = glob2local
    
    def train_sub_batches(self, model, patches, label_patches, out_patches, patch_features, weight_patches, sub_backward, retain_graph, level):
        
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
            "patches": tuple([patches_var.to(self.device)] + patch_features) if self.glob2local else patches_var.to(self.device),
            "previous_prediction": out_patches_var.to(self.device),
            "level": level
        }
        local_predictions, aggre_predictions = model(**params)
        
        # Calculate loss if current training level
        if sub_backward:
            
            loss = (1 - self.lamb_reg) * self.criterion(aggre_predictions, label_patches_var.to(self.device), weight_patches_var.to(self.device)) \
                + self.lamb_reg * self.reg_loss_fn(aggre_predictions, out_patches_var.to(self.device))
            if self.supervision and local_predictions is not None:
                loss += self.criterion(local_predictions[0], label_patches_var.to(self.device), None)
            loss.backward(retain_graph=retain_graph)

        # Update back to output_patches
        aggre_predictions.require_grad = False
        aggre_predictions = aggre_predictions.detach()
        if self.warping:
            aggre_predictions = extract_patch_from_warps(aggre_predictions)
        return local_predictions[1], aggre_predictions, loss
        
    def train_one_level(self, model, patch_size, images, labels, weights, out, out_features, sub_backward, training_level):
        
        # Create local patches
        patches, coordinates, templates, _, ratios, n_x, n_y, step_x, step_y = slice(images, patch_size, tanh_warping=self.warping)
        patches = map_parallel(lambda p_list: resize(p_list, self.rescale_size), patches)
        
        # Create local labels
        label_patches = slice_only(labels, patch_size,  n_x, n_y, step_x, step_y)
        
        # Create local previous prediction
        out_patches = slice_only(out, patch_size, n_x, n_y, step_x, step_y)
        #out_patches = list(map(lambda out_list: F.interpolate(torch.stack(out_list, dim=0), size=self.rescale_size, mode='bilinear'), out_patches))
        
        # Create local weights
        weight_patches = slice_only(weights.unsqueeze(1), patch_size, n_x, n_y, step_x, step_y)
        n = len(out_patches)
        temp = map_parallel(lambda out_list: F.interpolate(torch.stack(out_list, dim=0), size=self.rescale_size, mode='bilinear'), weight_patches + out_patches)
        weight_patches = temp[:n]
        out_patches = temp[n:]
        
        temp_features = [None] * len(images)
        
        for i in range(len(images)):
            j = 0
            while j < len(coordinates[i]):
                
                patch_features = crop_global_features(out_features[i], coordinates[i][j: j + self.sub_batch_size], ratios[i])
                temp_patch_features, patch_predictions, loss = self.train_sub_batches(model, \
                                                                patches[i][j : j+self.sub_batch_size], \
                                                                label_patches[i][j : j+self.sub_batch_size], \
                                                                out_patches[i][j : j+self.sub_batch_size], \
                                                                patch_features, \
                                                                weight_patches[i][j : j+self.sub_batch_size], \
                                                                sub_backward, \
                                                                not(i == len(images) - 1 and j + self.sub_batch_size >= len(coordinates[i])), \
                                                                training_level)
                # Save the patch features to restore
                if temp_features[i] is None:
                    temp_features[i] = temp_patch_features.detach_cpu()
                else:
                    temp_features[i] += temp_patch_features.detach_cpu()
                # Update the output
                out_patches[i][j : j+self.sub_batch_size] = patch_predictions.detach().to("cpu")
                j += self.sub_batch_size
                
            # Update output_patches back to out
        out = torch.stack(map_parallel(lambda i: patches2global(out_patches[i], out.shape[1:], patch_size, coordinates[i], templates[i]), range(len(images))), dim=0)
        #out = torch.stack(list(map(lambda i: patches2global(out_patches[i], out.shape[1:], patch_size, coordinates[i], templates[i]), range(len(images)))), dim=0)
        
        # Update features back
        temp_features = map_parallel(lambda i: temp_features[i].patches2global(coordinates[i], ratios[i]), range(len(images)))
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        out_feat = reduce((lambda x, y: x + y), temp_features)
                
        return out_feat.to(self.device), out, loss
            
    
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
        
        global_out, features = model(**params)
        
        # If training global only
        if self.training_level == -1:
            self.optimizer.zero_grad()
            loss = self.criterion(global_out, labels_glb.to(self.device))
            loss.backward()
            self.optimizer.step()
            return loss
        
        # Refine result with local branches
        out = F.interpolate(global_out, size=self.origin_size, mode='bilinear').to("cpu")
        
        # For each local branch
        for level in range(self.training_level + 1):
            features, out, loss = self.train_one_level(model, (self.patch_sizes[level], self.patch_sizes[level]), images, labels, weights, out, features, (level == self.training_level), level)

        # Return loss of last layer
        return loss         

class Evaluator(object):
    
    def __init__(self, device, sub_batch_size, eval_level, num_classes, patch_sizes, rescaled_size, origin_size, rgb2class, warping, glob2local):
        super(Evaluator, self).__init__()
        
        self.device = device
        self.eval_level = eval_level
        self.metric = ConfusionMatrix(num_classes)
        self.local_metric = ConfusionMatrix(num_classes)
        self.rescale_size = rescaled_size
        self.origin_size = origin_size
        self.patch_sizes = patch_sizes
        self.rgb2class = rgb2class
        self.sub_batch_size = sub_batch_size
        self.warping = warping
        self.glob2local = glob2local
    
    def infer_sub_batches(self, model, patches, out_patches, patch_features, level):
        
        # Refine by the local branch
        patches_var = images_transform(patches).to(self.device)
        
        out_patches_var = out_patches
        out_patches_var = F.interpolate(out_patches_var, size=self.rescale_size, mode='bilinear')
    
        params = {
            "mode": "local",
            "patches": tuple([patches_var.to(self.device)] + patch_features) if self.glob2local else patches_var.to(self.device),
            "previous_prediction": out_patches_var.to(self.device),
            "level": level
        }
        local_predictions, aggre_predictions = model(**params)
        
        if self.warping:
            local_predictions = extract_patch_from_warps(local_predictions[0])
            aggre_predictions = extract_patch_from_warps(aggre_predictions)

        return local_predictions[1], local_predictions[0], aggre_predictions
        
    def infer_one_level(self, model, patch_size, images, out, out_features, training_level):
            
        # Create local patches
        patches, coordinates, templates, _, ratios, n_x, n_y, step_x, step_y = slice(images, patch_size, tanh_warping=self.warping)
        patches = map_parallel(lambda p_list: resize(p_list, self.rescale_size), patches)
        
        # Create local previous prediction
        out_patches = slice_only(out, patch_size,  n_x, n_y, step_x, step_y)
        out_patches = map_parallel(lambda out_list: F.interpolate(torch.stack(out_list, dim=0), size=self.rescale_size, mode='bilinear'), out_patches)
        
        out_local = out.clone()
        out_local_patches= slice_only(out_local, patch_size,  n_x, n_y, step_x, step_y)
        out_local_patches = map_parallel(lambda out_list: F.interpolate(torch.stack(out_list, dim=0), size=self.rescale_size, mode='bilinear'), out_local_patches)
        
        temp_features = [None] * len(images)
        for i in range(len(images)):
            j = 0
            while j < len(coordinates[i]):
                patch_features = crop_global_features(out_features[i], coordinates[i][j: j + self.sub_batch_size], ratios[i])
                temp_patch_features, local_predictions, patch_predictions = self.infer_sub_batches(model, \
                                                                                                    patches[i][j : j+self.sub_batch_size], \
                                                                                                    out_patches[i][j : j+self.sub_batch_size], \
                                                                                                    patch_features, \
                                                                                                    training_level)
                out_patches[i][j : j+self.sub_batch_size] = patch_predictions.to("cpu")
                out_local_patches[i][j : j+self.sub_batch_size] = local_predictions.to("cpu")
                j += self.sub_batch_size
                
                # Save the patch features to restore
                if temp_features[i] is None:
                    temp_features[i] = temp_patch_features.detach_cpu()
                else:
                    temp_features[i] += temp_patch_features.detach_cpu()
                
            # Update output_patches back to out
            out[i] = patches2global(out_patches[i], out.shape[1:], patch_size, coordinates[i], templates[i])
            out_local[i] = patches2global(out_local_patches[i], out.shape[1:], patch_size, coordinates[i], templates[i])
            # Update features back
            temp_features[i] = temp_features[i].patches2global(coordinates[i], ratios[i])
        
        out_feat = reduce((lambda x, y: x + y), temp_features)
                
        return out_feat.to(self.device), out_local, out
    
    def get_scores(self):
        self.metric.synchronize_between_processes()
        self.local_metric.synchronize_between_processes()
        return {'aggregate': self.metric.get_scores(), 'local': self.local_metric.get_scores()}

    def reset_metrics(self):
        self.metric.reset()
        self.local_metric.reset()
    
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
            global_out, features = model(**params)
            
            out = F.interpolate(global_out, size=self.origin_size, mode='bilinear').to("cpu")
            out_local = F.interpolate(global_out, size=self.origin_size, mode='bilinear').to("cpu")
            
            # For each local branch
            for level in range(self.eval_level + 1):
                features, out_local, out = self.infer_one_level(model, (self.patch_sizes[level], self.patch_sizes[level]), images, out, features, level)
            out = torch.softmax(out, dim=1).argmax(1).numpy()
            labels = sample['label'] # PIL images
            labels_npy = masks_transform(labels, self.rgb2class, numpy=True)
            self.metric.update(labels_npy, out)
            
            out_local = torch.softmax(out_local, dim=1).argmax(1).numpy()
            self.local_metric.update(labels_npy, out_local)
                
            return out_local, out
            
        