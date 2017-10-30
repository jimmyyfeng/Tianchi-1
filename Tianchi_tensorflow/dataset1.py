from torch.utils.data import  Dataset
import pandas as pd
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
def crop(img,mask):
    size=64
    span=32
    origin=img.shape
    cropedimg=np.zeros([size,size,size]).astype(np.float16)
    cropedmask=np.zeros([size,size,size]).astype(np.float16)
    offset=np.random.randint(-5,5,3)
    crop_center=np.array([48,48,48])+offset
    cropedimg=img[np.clip(crop_center[0]-32,0,origin[0]):np.clip(crop_center[0]+32,0,origin[0]),\
                  np.clip(crop_center[1]-32,0,origin[1]):np.clip(crop_center[1]+32,0,origin[1]),\
                  np.clip(crop_center[2]-32,0,origin[2]):np.clip(crop_center[2]+32,0,origin[2])]
    cropedmask=mask[np.clip(crop_center[0]-32,0,origin[0]):np.clip(crop_center[0]+32,0,origin[0]),\
                  np.clip(crop_center[1]-32,0,origin[1]):np.clip(crop_center[1]+32,0,origin[1]),\
                  np.clip(crop_center[2]-32,0,origin[2]):np.clip(crop_center[2]+32,0,origin[2])]
    return cropedimg,cropedmask
class LungDataset(Dataset):
    def __init__(self, root,augument=False,crop=True):
        self.root=root
        # self.imgfiles = glob(root+'/*.mhd')
        self.imgfiles= glob(root+'img_cubic/*.npy')
        self.maskfiles=glob(root+'mask_cubic/*.npy')
        self._len = len(self.imgfiles)
        self.crop=crop
        self.augument = augument
    def load(self,index):
        return np.load(self.imgfiles[index]),np.load(self.maskfiles[index])
    def __getitem__(self, index):
        img,mask= self.load(index)
        if self.augument:
            img,mask= augument(img,mask)
        if self.crop:
            img,mask=crop(img,mask)
        #print img.shape
        scale=64
        img_tensor = torch.from_numpy(img.astype(np.float32)).view(1,scale,scale,scale)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).view(1,scale,scale,scale)
        file=self.imgfiles[index]
        return img_tensor, mask_tensor, file
        
    def __len__(self):
        return self._len
