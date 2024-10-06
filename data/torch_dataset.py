import cv2 
import torch 
import numpy as np 
from torch.utils.data import Dataset 
from torchvision.transforms import transforms 
from typing import List
from PIL import Image


class MinMaxNormalize(object):
    def __call__(self, tensor):
        # Get the minimum and maximum values of the tensor
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        # Apply Min-Max normalization
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized_tensor
    
class Transform: 
    train_transform = transforms.Compose([
        transforms.Resize((256,256)), 
        transforms.ToTensor(), 
        MinMaxNormalize()
    ]) 

    val_transform = transforms.Compose([ 
        transforms.Resize((256,256)), 
        transforms.ToTensor(), 
        MinMaxNormalize()
    ])  


    
class DepthDataset(Dataset): 
    def __init__(self, images_tensor : List[torch.tensor], labels_tensor : List[torch.tensor]) -> None: 
        self.images_tensor = images_tensor  
        self.labels_tensor = labels_tensor 

    def __len__(self) -> int: 
        return len(self.images_tensor) 

    def read_pfm(self, file : str):
        with open(file, 'rb') as f:
            # Header
            header = f.readline().decode('utf-8').rstrip()
            color = header == 'PF'

            # Dimensions
            dim_line = f.readline().decode('utf-8').rstrip()
            width, height = map(int, dim_line.split())

            # Scale (endian and range)
            scale = float(f.readline().decode('utf-8').rstrip())
            endian = '<' if scale < 0 else '>'  # Little endian if scale is negative

            # Pixel data
            data = np.fromfile(f, endian + 'f')  # Read floats from file
            shape = (height, width, 3) if color else (height, width)

            # Reshape and flip the image vertically
            return np.reshape(data, shape)[::-1, ...]
        
    def min_max_scaler(self, tensor):
        # Hitung nilai minimum dan maksimum dari tensor input
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        # Terapkan Min-Max scaling
        scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # Scaling ke [0, 1]
        return scaled_tensor

    def __getitem__(self, index):
        image = self.images_tensor[index]
        pfm = self.labels_tensor[index]
        return image, pfm 
