#encoding:utf-8
from Classifier_3d import Classifier
import tensorflow as tf
import numpy as np
import os
import csv
from cysb import voxel_2_world,get_filename
from glob import glob
from ops import load
import SimpleITK as sitk
import pandas as pd  

def get_topn(arr,n):
    tmp=np.sort(arr)
    index=[]
    for i in range(n):
        kk=np.where(arr==tmp[-1-i])[0]
        index.append(kk[0])
    return index

class Tester():
    def __init__(self,restore_from,model_dir,img_dir,batch_size=1,topN=3,is_trainning=False):
        #restore_from：模型加载路径
        #model_dir：模型所在文件夹
        #img_dir：原始图像路径
        self.restore_from=restore_from
        self.model_dir=model_dir
        self.img_dir=img_dir
        self.batch_size=batch_size
        self.is_trainning=is_trainning
        self.topN=topN
        self.image_batch=tf.placeholder(tf.float32, shape=[self.batch_size, 48, 48,48, 1])
        self.net,self.sess=self.CreateNet()
        
    def CreateNet(self):
        net=Classifier({'data': self.image_batch},batch_size=self.batch_size,is_training= self.is_trainning)
        self.all_trainable =tf.trainable_variables()
        self.restore_var = tf.global_variables()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        
        sess.run(init)
        if self.restore_from is not None:
            loader = tf.train.Saver(var_list=self.restore_var)
            load(loader, sess, self.restore_from,self.model_dir)
        return net,sess
    def do_Class(self):
    #nodule and it's center list
        nodule_list=glob(self.img_dir+"nodule_test/*_nodule.npy")
        center_list=glob(self.img_dir+"nodule_test/*_center.npy")
        f=open(self.img_dir+"csv/val/submission.csv", "wa")
        prob = self.net.layers['result']
        for i,patient in enumerate(nodule_list):
            patient_id=patient.split('/')[-1].split('_')[0]
            print 'doing on',patient_id
            patient_center=get_filename(center_list,patient_id)
            bb=np.load(patient)
            aa=np.load(patient_center)
            length=bb.shape[0]
            result=np.zeros([length,2])
            for j in range(length):
                img=bb[j]
                img=img[np.newaxis,:,:,:,np.newaxis]
                feed_dict={self.image_batch:img}
                result[j]=prob.eval(feed_dict,session=self.sess)
            if length<self.topN:
                topN=length
            else:
                topN=self.topN
            index=get_topn(result[:,0],topN)
            probability=result[index,0]
            center_=aa[index]
            # print center_
            # world=voxel_2_world(center_[:,::-1],patient_id)
            try:
                world=voxel_2_world(center_[:,::-1],self.img_dir,patient_id)
                # print world
                for j in range(len(index)):
                    if probability[j]>0.5:
                        row=list(world[j])
                        row.append(probability[j])
                        row=[patient_id]+row
                        csv_writer = csv.writer(f, dialect = "excel")
                        csv_writer.writerow(row)
            except:
                print patient_id,length
            # print probability,center_[:,::-1]
            # break
            if i%20==0:
                print i," hava done" 
if __name__=='__main__':
    restore_from='./models'
    model_dir='stage2'
    img_dir='/home/x/dc/remote_file/data/TianChi/'
    test=Tester(restore_from,model_dir,img_dir,topN=3)
    test.do_Class()
    