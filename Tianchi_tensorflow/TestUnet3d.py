#encoding:utf-8
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import os
from glob import glob
from skimage import measure
import time
import array
import math
import cv2
from ops import load
from Unet3D_v1 import Unet3D
import tensorflow as tf
from cysb import get_ct,resample,normalize,cropBlocks,segment_HU_scan_frederic
from dataset import LungDataset
import torch

from torch.utils.data import DataLoader
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x

    
class Tester():
    def __init__(self,restore_from,model_dir,img_dir,save_dir,batch_size=1,is_trainning=False,is_save=True):
        #restore_from：模型加载路径
        #model_dir：模型所在文件夹
        #img_dir：原始图像路径
        #save_dir：疑似节点保存路径
        #is_save：是否保存切得疑似节点
        self.restore_from=restore_from
        self.model_dir=model_dir
        self.img_dir=img_dir
        self.save_dir=save_dir
        self.is_save=is_save
        self.batch_size=batch_size
        self.is_trainning=is_trainning
        self.image_batch=tf.placeholder(tf.float32, shape=[1, 64, 64,64, 1])
        self.net,self.sess=self.CreateSegNet()
    def CreateSegNet(self):
        net=Unet3D({'data': self.image_batch},batch_size=self.batch_size,is_training= self.is_trainning)    
        restore_var = tf.global_variables()    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
  
        if self.restore_from is not None:
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, self.restore_from,self.model_dir)
        return net,sess

    def do_seg(self,file_name):
        prob = self.net.layers['result']
        print "preparing data................................."
        img_arr,origin,spacing=get_ct(self.img_dir+file_name+'.mhd')
        # seged=segment_HU_scan_frederic(img_arr)
        img_new=normalize(img_arr)
        depth,height,width=img_new.shape
        blocks,indexs=cropBlocks(img_new)
        print " data prepared................................."
        probs=np.zeros(img_new.shape,dtype=np.float32)
        num=np.array(img_new.shape)/64
        off=np.array(img_new.shape)-64*num
        off_min=off/2
        print "doing on patient:",file_name
        for i in range(blocks.shape[0]):
            feed_dict={self.image_batch:blocks[i][np.newaxis,:,:,:,np.newaxis]} 
            probs[off_min[0]+indexs[i,0]*64:off_min[0]+indexs[i,0]*64+64,
                off_min[1]+indexs[i,1]*64:off_min[1]+indexs[i,1]*64+64,
                off_min[2]+indexs[i,2]*64:off_min[2]+indexs[i,2]*64+64,
                 ]=prob.eval(feed_dict,session=self.sess)[0,:,:,:,0]
            if i%50==0:
                print "doing with:",i
    
        # probs=probs*seged
        labels = measure.label(probs,connectivity=2)
        label_vals = np.unique(labels)   
        regions = measure.regionprops(labels)
        centers = []
        crops=[]
        bboxes=[]
        for prop in regions:
            B = prop.bbox
            if B[3]-B[0]>0 and B[4]-B[1]>0 and B[5]-B[2]>0 :
                z=int((B[3]+B[0])/2.0)
                y=int((B[4]+B[1])/2.0)
                x=int((B[5]+B[2])/2.0)
                centers.append(np.array([z,y,x]))
                bboxes.append(B)
        for idx,bbox in enumerate(bboxes):
            crop=np.zeros([48,48,48],dtype=np.float32)
            crop_center=centers[idx]
            min_margin=crop_center-24
            max_margin=crop_center+24-np.array(img_new.shape)
            for i in range(3):
                if min_margin[i]<0:
                    crop_center[i]=crop_center[i]-min_margin[i]
                if max_margin[i]>0:
                    crop_center[i]=crop_center[i]-max_margin[i]
            crop=img_new[int(crop_center[0]-24):int(crop_center[0]+24),\
                         int(crop_center[1]-24):int(crop_center[1]+24),\
                         int(crop_center[2]-24):int(crop_center[2]+24)]
            crops.append(crop)
        if self.is_save:
            np.save(self.save_dir+file_name+"_nodule.npy",np.array(crops))
            np.save(self.save_dir+file_name+"_center.npy",np.array(centers))
            
if __name__ == '__main__':
    restore_from='./models'
    img_dir='/home/x/dcsb/data/TianChi/train/'
    save_dir='/home/x/dcsb/data/TianChi/nodule_train/'
    model_dir="unet3d_v1"
    test=Tester(restore_from,model_dir,img_dir,save_dir)
    all_test=glob(test.img_dir + "*.mhd") 
    print all_test[:10]
    for patient in tqdm(all_test):
        file_name=patient.split('/')[-1][:-4]
        test.do_seg(file_name)