#encoding:utf-8
from data.util import augument,check_center,drop_zero,make_mask,crop,zero_normalize,normalize,load_ct
import random
import numpy as np
from glob import glob
import Queue
import pandas as pd
from config import opt
    
class BoostClsDataset(object):
    
    def __init__(self,nodule_list,background_list,pre_prob_path,ratio=5,augument=True,phase="train",crop_size=[40,40,40]):
        '''
        nodule_list:从训练样本上切下来的节点块以及分割网络找到的节点快，全部包含至少一个结点，大小64*64*64
        background_list:分割网络在样本上找到的非节点块，作为负样本，大小与正样本相同 
        ratio:训练时正负样本的比例，由于测试时负样本远远多于正样本，故需控制训练时的比例
            训练时，ratio=5，测试时，ratio=20，给更多的负样本
        augument：是否进行数据增强 
        phase：是训练阶段还是验证阶段，如果验证，则不进行数据增强以及随机shuffle
            如果是训练阶段，政府样本比例
        crop_size:送入分类网络训练的块大小
        '''
        self.phase=phase
        self.augument=augument if self.phase=="train" else False
        self.ratio=ratio+1 if self.phase=="train" else 2
        self.pre_prob_path=pre_prob_path
        self.nodule_list=nodule_list
        self.background_list=background_list
        self.pos_len=len(self.nodule_list)
        self.train_size=len(self.nodule_list)+len(self.background_list)#训练集的大小
        print "pos samples num: ",len(self.nodule_list)
        print "neg samples num: ",len(self.background_list)
        print "train size: ",self.train_size
        self.crop_size=crop_size
        self._len=self.pos_len*self.ratio*8 if self.phase=="train" else self.train_size# 希望一个epoch多跑几个
        #训练时当平均每个正样本呗用了5次后就将训练集shuffle一次，验证时不需要shuffle
        #self.shuffle_idx=pos_len*8 if self.phase=="train" else self._len
        #self.count=0

    def __getitem__(self,index):
        
    
        if index/8%self.ratio==0:
            img=np.load(self.nodule_list[index/8/self.ratio]).astype(np.float32)
            label=np.array([1])
            nodule_name=self.nodule_list[index/8/self.ratio].split('/')[-1]
            print self.nodule_list[index/8/self.ratio]
            pre_prob=np.load(self.pre_prob_path+nodule_name)
        else:
            neg_file=self.background_list[np.random.randint(0,len(self.background_list))]
            img=np.load(neg_file).astype(np.float32)
            label=np.array([0])
            nodule_name=neg_file.split('/')[-1]
            print neg_file
            pre_prob=np.load(self.pre_prob_path+nodule_name)
        zyx=np.random.randint(-3,3,3)
        center=np.array([32,32,32])+[zyx[0],zyx[1],zyx[2]]
        half=np.array(self.crop_size)/2
        img=img[center[0]-half[0]:center[0]+half[0],center[1]-half[1]:center[1]+half[1],center[2]-half[2]:center[2]+half[2]]
        if self.augument:
            types=index%8
            img=augument(img,type=types)
        img=normalize(img)
        return img, label,pre_prob
 
    def __len__(self):
        return self._len 

class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self,img_list,augment=True,crop_size=[64,64,64],randn=7):
        self.img_list=img_list
        self.augument=augment
        self.randn=randn
        self.nodule_num=1244*2
        self.crop_size=crop_size
        self.imgdeque=Queue.Queue(maxsize=100)
        self.maskdeque=Queue.Queue(maxsize=100)
        self.file_index=0
        print self.img_list[:10]
    def __getitem__(self, index):
        # print index,'ffffffffffffffffffffffffffffffffff'

        
        if self.imgdeque.empty():
            
            print index,self.file_index,'-----------------'
            if self.file_index==len(self.img_list):
                self.file_index=0
                random.shuffle(self.img_list)
            imgs,masks= self.load_image(self.file_index)
            self.file_index +=1

            #self.indexdeque.put(file_index)
            aaa=np.arange(0,len(imgs))
            random.shuffle(aaa)
            for i in range(len(imgs)):
                self.imgdeque.put(imgs[aaa[i]])
                self.maskdeque.put(masks[aaa[i]])
        img=self.imgdeque.get()
        mask=self.maskdeque.get()
        if self.augument:
            img,mask =augument(img,mask)
        
       
            
        return img,mask
        
    def load_image(self,index):
        image,origin,spacing = load_ct(self.img_list[index])
        label,nodule_centers,_,_=make_mask(self.img_list[index])
        types=np.random.random_sample()
        cubic_list=[]
        masks_list=[]
        for ii in range(len(nodule_centers)):
            rand=self.randn
            offset=np.random.randint(-1*rand,rand,3)
            #center_index=np.random.randint(len(nodule_centers))
            cubic,mask=crop(image,label,v_center=offset+nodule_centers[ii],crop_size=self.crop_size)              
            cubic_list.append(normalize(cubic))
            masks_list.append(mask)
            cubic_,mask_=crop(image,label,v_center=None,crop_size=self.crop_size)
            cubic_list.append(normalize(cubic_))
            masks_list.append(mask_)
        return cubic_list,masks_list
    def __len__(self):
        return self.nodule_num*100




    
class ClsDataset(object):
    
    def __init__(self,nodule_list,background_list,ratio=5,augument=True,phase="train",crop_size=[40,40,40]):
        '''
        nodule_list:从训练样本上切下来的节点块以及分割网络找到的节点快，全部包含至少一个结点，大小64*64*64
        background_list:分割网络在样本上找到的非节点块，作为负样本，大小与正样本相同 
        ratio:训练时正负样本的比例，由于测试时负样本远远多于正样本，故需控制训练时的比例
            训练时，ratio=5，测试时，ratio=20，给更多的负样本
        augument：是否进行数据增强 
        phase：是训练阶段还是验证阶段，如果验证，则不进行数据增强以及随机shuffle
            如果是训练阶段，政府样本比例
        crop_size:送入分类网络训练的块大小
        '''
        self.phase=phase
        self.augument=augument if self.phase=="train" else False
        self.ratio=ratio+1 if self.phase=="train" else 2
        self.nodule_list=nodule_list
        self.background_list=background_list
        self.pos_len=len(self.nodule_list)
        self.train_size=len(self.nodule_list)+len(self.background_list)#训练集的大小
        print "pos samples num: ",len(self.nodule_list)
        print "neg samples num: ",len(self.background_list)
        print "train size: ",self.train_size
        self.crop_size=crop_size
        self._len=self.pos_len*self.ratio*8 if self.phase=="train" else self.train_size# 希望一个epoch多跑几个
        #训练时当平均每个正样本呗用了5次后就将训练集shuffle一次，验证时不需要shuffle
        #self.shuffle_idx=pos_len*8 if self.phase=="train" else self._len
        #self.count=0

    def __getitem__(self,index):
        
    
        if index/8%self.ratio==0:
            img=np.load(self.nodule_list[index/8/self.ratio]).astype(np.float32)
            label=np.array([1])
        else:
            neg_file=self.background_list[np.random.randint(0,len(self.background_list))]
            img=np.load(neg_file).astype(np.float32)
            label=np.array([0])
        zyx=np.random.randint(-3,3,3)
        center=np.array([32,32,32])+[zyx[0],zyx[1],zyx[2]]
        half=np.array(self.crop_size)/2
        img=img[center[0]-half[0]:center[0]+half[0],center[1]-half[1]:center[1]+half[1],center[2]-half[2]:center[2]+half[2]]
        if self.augument:
            types=index%8
            img=augument(img,type=types)
        img=normalize(img)
        return img, label
 
    def __len__(self):
        return self._len 
    
class Dataset3(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self,img_path,augment=True,crop_size=[64,64,64],randn=20):
        self.img_list=glob(img_path+'*.npy')
        self.augument=augment
        self.randn=randn
        self.nodule_num=len(self.img_list)
        self.crop_size=crop_size
        self.file_index=0
        print self.img_list[:10]
    def __getitem__(self, index):

        
        
        if self.file_index==len(self.img_list):
            self.file_index=0
            random.shuffle(self.img_list)
        imgs,masks= self.load_image(self.file_index)
        self.file_index +=1
        if self.augument:
            imgs,masks =augument(imgs,masks)            
        return normalize(imgs),masks
        
    def load_image(self,index):
        cubic=np.load(self.img_list[index])
        masks=np.load(self.img_list[index].replace('train_nodule','train_nodule_mask'))
        zyx=np.random.randint(-1*self.randn,self.randn,3)
        center=np.array(cubic.shape)/2+[zyx[0],zyx[1],zyx[2]]
        half=[32,32,32]
        img=cubic[center[0]-half[0]:center[0]+half[0],center[1]-half[1]:center[1]+half[1],center[2]-half[2]:center[2]+half[2]]
        mask=masks[center[0]-half[0]:center[0]+half[0],center[1]-half[1]:center[1]+half[1],center[2]-half[2]:center[2]+half[2]]
        return img,mask
    def __len__(self):
        return self.nodule_num
    
if __name__=='__main__':
    from data.data.dataloader import DataLoader
    sample_list=open("/home/x/dcsb/Tianchi_caffe/train_cls.txt", 'r').read().splitlines()
    nodule_list=[r[:-2] for r in sample_list if r[-1]=='1']
    background_list=[r[:-2] for r in sample_list if r[-1]=='0']
    pre_prob_path='/mnt/7/train_prob_cls_res_val/'
    dataset=BoostClsDataset(nodule_list,background_list,pre_prob_path,ratio=2,augument=True,phase="train")
    dataloader = DataLoader(dataset,batch_size=1,       
                        num_workers=1,
                        shuffle=True,
                        drop_last=True
                        )
    for ii, (input, label,pre_prob) in enumerate(dataloader):
        print label,pre_prob
        if ii>10:break
        
    
