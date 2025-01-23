from PIL import Image
import numpy as np
import torch

class SRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, hr_images, scale_factor):
        self.hr_images = hr_images
        self.scale_factor = scale_factor
    def __len__(self):
        return len(self.hr_images)
    def __getitem__(self, index):
        hr_image = self.hr_images[index]
        hr_image = Image.fromarray(hr_image)
        hr_image = hr_image.resize(((hr_image.width//self.scale_factor)*self.scale_factor, (hr_image.height//self.scale_factor)*self.scale_factor), resample=Image.BICUBIC)

        lr_image = hr_image.resize((hr_image.width//self.scale_factor, hr_image.height//self.scale_factor), resample=Image.BICUBIC)  
        lr_image = lr_image.resize((lr_image.width*self.scale_factor, lr_image.height*self.scale_factor), resample=Image.BICUBIC)

        hr_image = np.array(hr_image, dtype="float32")
        hr_image/=255

        lr_image = np.array(lr_image, dtype="float32")
        lr_image/=255
        
        return torch.tensor(lr_image.T), torch.tensor(hr_image.T)