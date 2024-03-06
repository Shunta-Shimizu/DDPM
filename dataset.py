import torch
import torchvision
import glob
from PIL import Image

class DDPM_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, img_files, transform):
        self.data_path = data_path
        self.img_files = img_files
        self.transform = transform
    
    def __getitem__(self, index):
        img = Image.open(self.data_path + self.img_files[index])
        img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.img_files)
