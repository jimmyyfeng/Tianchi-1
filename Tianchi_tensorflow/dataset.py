from torch.utils.data import  Dataset
import pandas as pd
from cysb import make_mask_for_patients_V1
import os
import numpy as np
import torch
from glob import glob
def augument(imgs,masks):
    type=np.random.randint(0,4)
    if type==0:
        return imgs[::-1,:,:],masks[::-1,:,:]
    elif type==1:
        return imgs[:,::-1,:],masks[:,::-1,:]
    elif type==2:
        return imgs[:,:,::-1],masks[:,:,::-1]
    else:
        return imgs,masks
class LungDataset(Dataset):
    def __init__(self, root,augument=False):
        self.root=root
        self.imgfiles = glob(root+'/*.mhd')
        # self.maskfiles=list(set([file.split('.')[0] for file in os.listdir(root+'mask_cubic/')]))
        self._len = len(self.imgfiles)
        self.augument = augument
    def __getitem__(self, index):
        img,mask= make_mask_for_patients_V1(self.imgfiles[index])
        if self.augument:
            img,mask= augument(img,mask)
        #print img.shape
        scale=64
        img_tensor = torch.from_numpy(img.astype(np.float32)).view(1,scale,scale,scale)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).view(1,scale,scale,scale)
        file=self.imgfiles[index]
        return img_tensor, mask_tensor, file
        
    def __len__(self):
        return self._len
