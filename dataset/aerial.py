import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

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

    def __init__(self, root, image_names, label=False, transform=False):
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