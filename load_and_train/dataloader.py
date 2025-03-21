import os 
import torch 
import torchvision
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd 
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import tqdm
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

class polypSegmentationDataset(Dataset):
    def __init__(self, root, img_paths, mask_paths, transform=None):
        super(Dataset,self).__init__()
        self.root = root 
        self.image_paths = img_paths 
        self.mask_paths = mask_paths  
        self.transform = transform
        

        # self.mask_transform = transforms.Resize((28, 28), interpolation=Image.NEAREST)
        assert len(self.image_paths) == len(self.mask_paths)

        self.class_dict = {
            (0, 0, 0): 0,       
            (255, 255, 255): 1  
        }

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.image_paths[idx])
        mask_path = os.path.join(self.root, 'masks', self.mask_paths[idx])
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
        
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        
        if mask.ndim == 2:  
            mask = mask.unsqueeze(0)
        mask = mask.float()  
        
        return image, mask







image_file =r'Data/kvasir-seg/Kvasir-SEG/images'
mask_file = r'Data/kvasir-seg/Kvasir-SEG/masks'







    # def __getitem__(self, idx):
    #   
    #     if self.transform:
    #         image = self.transform(image)
    #         mask = self.transform(mask)
    #     else:
            
    #         mask = torch.tensor(np.array(mask), dtype=torch.long)

        

    #     return image, mask



    #  image = cv2.imread(self.image_paths[idx])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     if not os.path.exists(image):
    #         raise FileNotFoundError(f"Image not found: {image}")

    #     mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
    #     augmented = self.transform(image=image, mask=mask)








        #  img_path = os.path.join(self.root, 'images', self.image_paths[idx])
        # mask_path = os.path.join(self.root, 'masks', self.mask_paths[idx])
        
        
        # image = cv2.imread(img_path)
        # if image is None:
        #     raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)

        
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # if mask is None:
        #     raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")
        # mask = Image.fromarray(mask)
        # augmented = self.transform(image=image, mask=mask)

        
        
       
        # image = augmented["image"]
        # mask = augmented["mask"]
        
        # return image, mask
