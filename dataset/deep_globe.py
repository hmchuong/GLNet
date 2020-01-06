import os
import random

import cv2
import numpy as np
from PIL import Image, ImageFile

import torch
import torch.utils.data as data

from torchvision.transforms import ToTensor
from torchvision import transforms

from .base import BaseDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DeepGlobe(BaseDataset):
    """input and label image dataset"""

    def __init__(self, root, ids, label=False, transform=False):
        super(DeepGlobe, self).__init__()
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.classdict = {1: "urban", 2: "agriculture", 3: "rangeland", 4: "forest", 5: "water", 6: "barren", 0: "unknown"}
        self.label2color = {
            0: [0, 0, 0],
            1: [0, 255, 255],
            2: [255, 255, 0],
            3: [255, 0, 255],
            4: [0, 255, 0],
            5: [0, 0, 255],
            6: [255, 255, 255]
        }
    
    @staticmethod
    def prepare_subset_ids(data_path):
        ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train", "Sat")) if is_image_file(image_name)]
        ids_val = [image_name for image_name in os.listdir(os.path.join(data_path, "crossvali", "Sat")) if is_image_file(image_name)]
        ids_test = [image_name for image_name in os.listdir(os.path.join(data_path, "test", "Sat")) if is_image_file(image_name)]
        return (ids_train, os.path.join(data_path, "train")), (ids_val, os.path.join(data_path, "crossvali")), (ids_test, os.path.join(data_path, "test"))

    def __getitem__(self, index):
        sample = {}
        sample['id'] = self.ids[index][:-8]
        image = Image.open(os.path.join(self.root, "Sat/" + self.ids[index])) # w, h
        sample['image'] = image
        if self.label:
            label = Image.open(os.path.join(self.root, 'Label/' + self.ids[index].replace('_sat.jpg', '_mask.png')))
            sample['label'] = label
        if self.transform and self.label:
            image, label = self._transform(image, label)
            sample['image'] = image
            sample['label'] = label
            
        # Generate weights from label
        label = np.array(sample['label'])
        laplacian_label = cv2.Laplacian(label, cv2.CV_64F)
        laplacian_label = (np.sum(laplacian_label.astype(np.uint8), axis=2) > 0).astype(np.float32)
        laplacian_label[laplacian_label == 0] = 0.5
        sample['weight'] = torch.from_numpy(laplacian_label)
        
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
        return len(self.ids)