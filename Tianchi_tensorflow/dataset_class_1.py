from torch.utils.data import  Dataset
import pandas as pd
import os
import numpy as np
import torch
from glob import glob
def augument(imgs):
    type=np.random.randint(0,5)
    if type==0:
        return imgs[::-1,:,:]
    elif type==1:
        return imgs[:,::-1,:]
    elif type==2:
        return imgs[:,:,::-1]
    elif type==3:
        return imgs+0.001*np.random.randn(48,48,48)
    else:
        return imgs
class LungDataset(Dataset):
    def __init__(self, root,augument=False):
        self.root=root
        self.imgfiles = glob(root+'/class_nodule/nodule*.npy')
        self.labelfiles=glob(root+'/class_nodule/label*.npy')
        self._len_img = len(self.imgfiles)
        self._len=1750
        self.augument = augument
    def __getitem__(self, index):
        ith=np.random.randint(0,self._len_img,1)[0]
        nodules=np.load(self.imgfiles[ith])
        labels=np.load(self.labelfiles[ith])
        Num=nodules.shape[0]
        idx=np.random.randint(0,Num,1)[0]
        img=nodules[idx]
        label=labels[idx]
        if self.augument:
            img= augument(img)
        scale=48
        try:
            img_tensor = torch.from_numpy(img.copy()).view(1,scale,scale,scale)
        except: 
            self.__getitem__(self,index+1)
        
        return img_tensor, label
        
    def __len__(self):
        return self._len
