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
    label = (label * 255).astype(np.uint8)
    return np.stack([label, label, label], axis=2)


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


class IDRID(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, dataframe, global_size, label=False, transform=False):
        super(IDRID, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.dataframe = dataframe
        
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        self.resizer = transforms.Resize((3410, 3410))
        self.transformer = transforms.Compose([
            transforms.Resize(global_size, Image.BILINEAR),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        sample = {}
        data_row = self.dataframe.loc[index]
        sample['id'] = os.path.basename(data_row["image"]).replace(".png", "")
        image = Image.open(os.path.join(self.root, data_row["image"])) # w, h
        sample['image'] = transforms.functional.resize(image, (3410, 3410))
        if self.label:
            if str(data_row["mask"]) == 'nan':
                label = Image.fromarray(np.zeros((3410, 3410), dtype=np.uint8))
            else:
                label = Image.open(os.path.join(self.root, data_row['mask']))
                label = transforms.functional.resize(label, (3410, 3410))
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
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}
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
        return len(self.dataframe)