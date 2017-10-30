#coding:utf-8
import caffe
from data.dataloader import DataLoader
import numpy as np
from PIL import Image
from data.dataset import ClsDataset
import random
from util import load_ct,make_mask,crop,normalize
class ClsDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        """
        
        phase:train  or  val  or  test
        batch_size:batch_size
        是否shuffle:是否shuffle
        sample_list：全部样本列表，包括正样本和负样本
        augument：是否数据增强
        nodule_list:正样本
        background_list:负样本
        crop_size:块大小
        
        """
        
        params = eval(self.param_str)
        self.data_root=params['data_root']#txt 文件保存父路径
        self.phase=params['phase']#train or val 
        if self.phase  not in ['train','test','val']:
            raise Exception("phase must be train  or  val  or  test")
        self.sample_list=open(self.data_root+self.phase+"_cls.txt", 'r').read().splitlines()
        self.ratio=params['ratio'] if self.phase=='train' else len(self.sample_list)
        self.augument=params.get('augument', True) if self.phase=='train' else False
        self.shuffle=True  if self.phase=='train' else False
        self.nodule_list=[r[:-2] for r in self.sample_list if r[-1]=='1']
        self.background_list=[r[:-2] for r in self.sample_list if r[-1]=='0']
        self.crop_size=params.get('crop_size', [40,40,40])
        self.batch_size =params.get('batch_size', 16)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        top[0].reshape(
            self.batch_size, 1, *self.crop_size)
        top[1].reshape(self.batch_size, 1)
        
        self.dataset=ClsDataset(self.nodule_list,self.background_list,ratio=self.ratio,augument=self.augument,phase=self.phase)
        self.dataloader = iter(DataLoader(self.dataset,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=1,drop_last=True))
        
       
       
    def forward(self,bottom, top):
        """
        Load data.
        """
        
        top[0].data[:,0],top[1].data[:] = self.dataloader.next()    

    def reshape(self, bottom, top):
        pass
       
    
        