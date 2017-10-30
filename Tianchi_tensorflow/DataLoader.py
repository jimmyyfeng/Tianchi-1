from torch.utils.data import  Dataset
import pandas as pd
import Queue
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
        self._len = 1244*2
        self.indexx=0
        self.file_num=len(self.imgfiles)
        self.augument = augument
        self.imgdeque=Queue.Queue(maxsize=100)
        self.maskdeque=Queue.Queue(maxsize=100)
    def __getitem__(self, index):
        if self.imgdeque.empty():
            img,mask= make_mask_for_patients_V1(self.imgfiles[self.indexx])
            self.indexx=self.indexx+1
            if self.indexx==self.file_num:
                self.indexx=0
            for i in range(len(img)):
                self.imgdeque.put(img[i])
                self.maskdeque.put(mask[i])

        img=self.imgdeque.get()
        mask=self.maskdeque.get()
        if self.augument:
            img,mask= augument(img,mask)
        #print img.shape
        scale=64
        img_tensor = torch.from_numpy(img.astype(np.float32)).view(1,scale,scale,scale)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).view(1,scale,scale,scale)
        file=self.imgfiles[self.indexx]
        return img_tensor, mask_tensor, file
        
    def __len__(self):
        return self._len
