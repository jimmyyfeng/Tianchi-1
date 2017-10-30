#coding:utf-8
import caffe
from data.dataloader import DataLoader
import numpy as np
from PIL import Image
from data.dataset import Dataset as Dataset
import random
from util import load_ct,make_mask,crop,normalize
from tqdm import tqdm
class SegDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        """
        img_root:以及图片 存放路径 train.txt,val.txt,test.txt存放路径
        phase:train  or  val  or  test
        batch_size:batch_size
        random:是否shuffle
        
        """
        
        params = eval(self.param_str)
        self.img_root=params['img_root']
        self.phase=params['phase']
        self.random = params.get('randomize', True)
        self.crop_size=params.get('crop_size', [48,48,48])
        self.batch_size = params['batch_size']
        if self.phase  not in ['train','test','val']:
            raise Exception("phase must be train  or  val  or  test")        
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        self.data = {}   
        top[0].reshape(
            self.batch_size, 1, *self.crop_size)
        top[1].reshape(self.batch_size, 1,*self.crop_size)
        # load indices for images and labels
        split_f  = '{}txtfiles/{}.txt'.format(self.img_root,self.phase)
        self.indices = open(split_f, 'r').read().splitlines()
        #self.dataset=Dataset(self.indices,augment=True,crop_size=self.crop_size,randn=15)
        self.dataset=Dataset(self.indices,augment=True,crop_size=self.crop_size,randn=7)
        self.dataloader = iter(DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=1,drop_last=True))
        #import ipdb;ipdb.set_trace()
        self.idx = 0
        self.load_image=load_ct
        # make eval deterministic
        if 'train' not in self.phase:
            self.random = False 
        # randomization: seed and pick
        if self.random:
            random.shuffle(self.indices)
    def forward(self,bottom, top):
        """
        Load data.
        """
        #for ii in range(self.batch_size):
        #    im,label=self.load_next_data()
        #    top[0].data[ii,0] = im
        #    top[1].data[ii,0] = label
        top[0].data[:,0],top[1].data[:,0] = self.dataloader.next()    
#    def load_next_data(self):
#      
#        image,origin,spacing = self.load_image(self.indices[self.idx])
#        label,nodule_centers,_,_=make_mask(self.indices[self.idx])
#        types=np.random.random_sample()
#        if types>0.5:
#            center_index=np.random.randint(len(nodule_centers))
#            cubic,mask=crop(image,label,v_center=nodule_centers[center_index],crop_size=self.crop_size)              
#        else:
#            cubic,mask=crop(image,label,v_center=None,crop_size=self.crop_size)
#        self.idx +=1
#        if self.idx == len(self.indices):
#            self.idx = 0
#            random.shuffle(self.indices)
#        return  normalize(cubic),mask
    def reshape(self, bottom, top):
        pass
        #print "reshaping....."
        #image,label=self.load_next_data()
        #self.data = image
        #self.label = label
        # reshape tops to fit (leading 1 is for batch dimension)
        #top[0].reshape(self.batch_size, 1,*self.data.shape[1:])
        #top[1].reshape(self.batch_size, 1,*self.label.shape[1:])
    
        