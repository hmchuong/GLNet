import os
import random

import cv2
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision.transforms import ToTensor
from torchvision import transforms
import torchvision.transforms.functional as TF

import utils.log as track
from .base import BaseDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Aerial(BaseDataset):
    """input and label image dataset"""

    def __init__(self, root, image_names, label=False, transform=False):
        super(Aerial, self).__init__()
        self.root = root
        self.label = label
        self.transform = transform
        self.image_names = image_names
        self.image_size = 5000
        self.label2color = {0: [0, 0, 0], 1: [255, 255, 255]}
    
    @staticmethod
    def prepare_subset_ids(data_path):
        image_names = [name for name in os.listdir(os.path.join(data_path, "images")) if name.endswith(".tif")]
        train_names, val_test_names = train_test_split(image_names, train_size=126, random_state=2019)
        val_names, test_names = train_test_split(val_test_names, train_size=0.5, random_state=2019)
        return train_names, val_names, test_names

    def __getitem__(self, index):
        sample = {}
        image_name = self.image_names[index]
        sample['id'] = image_name.replace(".tif", "")
        image = Image.open(os.path.join(self.root, "images", image_name)).convert("RGB") # w, h
        sample['image'] = transforms.functional.resize(image, (self.image_size, self.image_size))
        
        if self.label:
            label = Image.open(os.path.join(self.root, "gt", image_name)).convert("RGB")
            label = transforms.functional.resize(label, (self.image_size, self.image_size))
            if self.transform:
                image, label = self._transform(sample['image'], label)
                sample['image'] = image
            sample['label'] = label
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