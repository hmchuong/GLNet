import torch.utils.data as data
from torchvision.transforms import ToTensor
import numpy as np

class BaseDataset(data.Dataset):
    
    def __init__(self):
        self.label2color = {}
        pass

    def classToRGB(self, label):
        l, w = label.shape[0], label.shape[1]
        colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
        for classnum, color in self.label2color.items():
            indices = np.where(label == classnum)
            colmap[indices[0].tolist(), indices[1].tolist(), :] = color
        transform = ToTensor()
        return transform(colmap)
    
    def RGBToClass(self, label):
        l, w = label.shape[0], label.shape[1]
        classmap = np.zeros(shape=(l, w))
        for classnum, color in self.label2color.items():
            indices = np.where(np.all(label == tuple(color), axis=-1))
            classmap[indices[0].tolist(), indices[1].tolist()] = classnum
        return classmap
    
    @staticmethod
    def prepare_subset_ids(data_path):

        pass