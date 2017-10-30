from torch.utils.data import  Dataset
import pandas as pd
from cysb import make_mask_for_patients_V1
import os
import Queue
import numpy as np
import torch
from glob import glob
def augument(imgs):
    type=np.random.randint(0,4)
    if type==0:
        return imgs[::-1,:,:]
    elif type==1:
        return imgs[:,::-1,:]
    elif type==2:
        return imgs[:,:,::-1]
    else:
        return imgs
class LungDataset(Dataset):
    def __init__(self, root,augument=False):
        self.root=root
        self.imgfiles = glob(root+'/train/*.mhd')
        self.posfiles = glob(root+'/class_nodule/*_1.npy')
        self.negfiles = glob(root+'/class_nodule/*_0.npy')
        self._len = 1244*2+2*len(self.posfiles)
        self.indexx=0
        self.file_num=len(self.imgfiles)
        self.augument = augument
        self.imgdeque=Queue.Queue(maxsize=100)
        self.maskdeque=Queue.Queue(maxsize=100)
    def __getitem__(self, index):
        types=np.random.randint(0,2)
        if types==0:
            if self.imgdeque.empty():
                img,mask= make_mask_for_patients_V1(self.imgfiles[self.indexx],a=0,width=48,rand=1,data_dir=self.root+'/')
                self.indexx=self.indexx+1
                if self.indexx==self.file_num:
                    self.indexx=0
                for i in range(len(img)):
                    self.imgdeque.put(img[i])
                    self.maskdeque.put(mask[i])
    
            img=self.imgdeque.get()
            mask=self.maskdeque.get()
            if np.sum(mask)>8:
                label=np.array([1,0])
            else:
                label=np.array([0,1])
        else:
            zeroOrone=np.random.randint(0,2)
            if zeroOrone==1:
                img=np.load(self.posfiles[np.random.randint(0,len(self.posfiles))])
                label=np.array([1,0])
            else:
                img=np.load(self.negfiles[np.random.randint(0,len(self.negfiles))])
                label=np.array([0,1])
        if self.augument:
            img= augument(img)
        scale=48
        try:
            img_tensor = torch.from_numpy(img.copy()).view(1,scale,scale,scale)
        except:
            self.__getitem__(self,index+1)
        file=self.imgfiles[self.indexx-1]
        return img_tensor, label, file
        
    def __len__(self):
        return self._len
