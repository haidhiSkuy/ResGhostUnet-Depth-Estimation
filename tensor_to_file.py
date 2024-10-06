import os 
import cv2 
import glob 
import torch
import numpy as np 
from tqdm import tqdm
from PIL import Image  
import io
from data.torch_dataset import Transform 

def read_pfm(file : str):
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
        
def min_max_scaler(tensor):
    # Hitung nilai minimum dan maksimum dari tensor input
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    # Terapkan Min-Max scaling
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # Scaling ke [0, 1]
    return scaled_tensor


dataset_root = "/mnt/d/backup/InSpaceType_all_sp8/home/InSpaceType" 
dataset = []
for seq in os.listdir(dataset_root): 
    sequence = os.path.join(dataset_root, seq) 

    images = sorted(glob.glob(sequence + "/*.jpg"))
    pmfs = sorted(glob.glob(sequence + "/*.pfm"))  

    for image, pmf in zip(images, pmfs): 
        dataset.append({'image':image, 'label':pmf})

image_tensors = [] 
pfm_tensors = []
for dat in tqdm(dataset): 
    image = dat["image"]
    image = Image.open(image)
    transformed = Transform.train_transform(image)
    image_tensors.append(transformed)

    label = dat["label"]
    pfm_image = read_pfm(label) 
    pfm_image = cv2.resize(pfm_image, (256,256))
    pfm_tensor = torch.from_numpy(pfm_image)
    pfm_tensor = min_max_scaler(pfm_tensor) 
    pfm_tensors.append(pfm_tensor) 

print("Saving to file")  
buffer_image = io.BytesIO()
torch.save(image_tensors, buffer_image) 

buffer_pfm = io.BytesIO()
torch.save(pfm_tensors, buffer_pfm) 