import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import utils.log as track

ImageFile.LOAD_TRUNCATED_IMAGES = True


def classToRGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
    indices = np.where(label == 1)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 0)
    colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    transform = ToTensor()
    return transform(colmap)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(7):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=7, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


class Aerial(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, image_names, global_size, label=False, transform=False):
        super(Aerial, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.image_names = image_names
        self.image_size = 5000
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        self.resizer = transforms.Resize((self.image_size, self.image_size))
        self.transformer = transforms.Compose([
            transforms.Resize(global_size, Image.BILINEAR),
            transforms.ToTensor()
        ])
        
    def __getitem__(self, index):
        sample = {}
        image_name = self.image_names[index]
        sample['id'] = image_name.replace(".tif", "")
        image = Image.open(os.path.join(self.root, "images", image_name)).convert("RGB") # w, h
        sample['image'] = transforms.functional.resize(image, (self.image_size, self.image_size))
        
        if self.label:
            label = Image.open(os.path.join(self.root, "gt", image_name)).convert("RGB")
            label = transforms.functional.resize(label, (self.image_size, self.image_size))
            label = np.array(label)[:,:,0]
            label[label > 0] = 1
            label[label == 0] = 0
            label = Image.fromarray(label)
            if self.transform:
                image, label = self._transform(sample['image'], label)
                sample['image'] = image
            sample['label'] = label
            sample['label_npy'] = np.array(label)
        sample['image_glb'] = self.transformer(sample['image'])
        return sample

    def _transform(self, image, label):

        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)
        
        return image, label


    def __len__(self):
        return len(self.image_names)
        
class AerialSubdatasetMode2(data.Dataset):
    def __init__(self, images_glb, ratios, coordinates, patches, label_patches, p_size):
        super(AerialSubdatasetMode2, self).__init__()
        self.images_glb = images_glb
        self.ratios = ratios
        self.coordinates = coordinates
        self.patches = patches
        self.label_patches = label_patches
        self.p_size = p_size
        
    def __getitem__(self, index):
        image_idx = index // len(self.coordinates[0])
        coord_idx = index % len(self.coordinates[0])
        sample = {}
        patch = TF.to_tensor(self.patches[image_idx][coord_idx])
        sample['id'] =image_idx
        sample['patch'] = patch
        coord = self.coordinates[image_idx][coord_idx]
        sample['coord'] = coord
        label = TF.resize(self.label_patches[image_idx][coord_idx], self.p_size, interpolation=Image.NEAREST)
        label = torch.from_numpy(np.array(label).astype('int32')).long()
        sample['label'] = label
        
        sample['image_glob'] = self.images_glb[image_idx: image_idx + 1]
        sample['ratio'] = self.ratios[image_idx]
        sample['n_patch'] = len(self.coordinates[image_idx])
        
        return sample
        
    def __len__(self):
        return sum([len(x) for x in self.coordinates])

class AerialSubdatasetMode3a(data.Dataset):
    def __init__(self, patches, coordinates, images_glb, ratios, templates):
        super(AerialSubdatasetMode3a, self).__init__()
        self.patches = patches
        self.coordinates = coordinates
        self.images_glb = images_glb
        self.ratios = ratios
        self.templates = templates
        
    def __getitem__(self, index):
        image_idx = index // len(self.coordinates[0])
        coord_idx = index % len(self.coordinates[0])
        sample = {}
        patch = TF.to_tensor(self.patches[image_idx][coord_idx])
        sample['id'] = image_idx
        sample['coord_id'] = coord_idx
        sample['patch'] = patch
        
        coord = self.coordinates[image_idx]
        sample['image_glob'] = self.images_glb[image_idx: image_idx + 1]
        sample['ratio'] = self.ratios[image_idx]
        sample['coord'] = coord
        sample['template'] = self.templates[image_idx]
        return sample
        
    def __len__(self):
        return sum([len(x) for x in self.coordinates])

class AerialSubdatasetMode3b(data.Dataset):
    def __init__(self, label_patches, p_size, fm_patches, coordinates, ratios):
        super(AerialSubdatasetMode3b, self).__init__()
        self.label_patches = label_patches
        self.p_size = p_size
        self.fm_patches = fm_patches
        self.coordinates = coordinates
        self.ratios = ratios
        
    def __getitem__(self, index):
        image_idx = index // len(self.coordinates[0])
        coord_idx = index % len(self.coordinates[0])
        sample = {}
        label = TF.resize(self.label_patches[image_idx][coord_idx], self.p_size, interpolation=Image.NEAREST)
        label = torch.from_numpy(np.array(label).astype('int32')).long()
        sample['id'] = image_idx
        sample['label'] = label
        sample['fl'] = self.fm_patches[image_idx][coord_idx]
        coord = self.coordinates[image_idx][coord_idx]
        sample['ratio'] = self.ratios[image_idx]
        sample['coord'] = coord
        return sample
        
    def __len__(self):
        return sum([len(x) for x in self.coordinates])