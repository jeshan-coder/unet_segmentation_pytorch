from torch.utils.data import Dataset,DataLoader
import os
from torchvision.transforms.v2 import Compose,ToTensor,Normalize
from PIL import Image
import numpy as np
class UnetDataPreparation(Dataset):
    def __init__(self,image_path,mask_path):
        super().__init__()
            
        self.image_path=image_path
        self.mask_path=mask_path
            
            
        #sort images
            
        self.images=os.listdir(self.image_path)
        self.images=sorted(self.images)
            
            
        #sort masks
            
        self.masks=os.listdir(self.mask_path)
        self.masks=sorted(self.masks)
            
            
        #transformation
            
    
    
    def __len__(self):
        
        return len(self.masks)
    
    
    def __getitem__(self,index):
        
        
        # get image name
        image=self.images[index]
        mask=self.masks[index]
        
        #resize images
        
        # open imaages
        image=Image.open(self.image_path+image)
        mask=Image.open(self.mask_path+mask)
        
        #resize image and mask
        image=image.resize((256,256))
        mask=mask.resize((256,256))
        
        
        #calculate mean and std
        mean, std = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        
        #transformer initialization
        
        #image
        transformer_image=Compose([
                ToTensor(),
                #Normalize(mean=mean, std=std),
                ])
        
        #mask
        transformer_mask=Compose([
                ToTensor(),
                ])
        
        #convert image to rgb only
        image=image.convert("RGB")
        
        #convert mask to gray value only (not necessary)
        mask=mask.convert('L')
        
        #convert image and mask to torch
        image=transformer_image(image)
        mask=transformer_mask(mask)
        
        return image,mask

        
        
        
        
        
        
        
        