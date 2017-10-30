import argparse
from datetime import datetime
import os
import sys
import time
import random
from Unet3D_v1 import Unet3D
import tensorflow as tf
from util import Visualizer
import numpy as np
from DataLoader import LungDataset
import torch
import torchnet as tnt
from ops import mse,normalize,dice_coef_loss,extraLoss,extraLoss1,pixelwise_cross_entropy,save,load
from torch.utils.data import DataLoader
class Trainer():
    def __init__(self,restore_from,model_dir,logs_dir,img_dir,batch_size = 2,max_run = 1000):
        self.restore_from=restore_from
        self.model_dir=model_dir
        self.logs_dir=logs_dir
        self.img_dir=img_dir
        self.batch_size=batch_size
        self.max_run=max_run
        self.vis=Visualizer()
        self.global_steps=0
        self.image_batch=tf.placeholder(tf.float32, shape=[None, 64, 64,64, 1])
        self.label_batch=tf.placeholder(tf.float32, shape=[None, 64, 64,64, 1])
        self.dice_loss_meter =tnt.meter.AverageValueMeter()
        self.build_model()
    def build_model(self):
        self.net=Unet3D({'data': self.image_batch},batch_size=self.batch_size)
        self.prob = self.net.layers['result']
        self.logits=self.net.layers['conv_8']
        self.cross_loss = pixelwise_cross_entropy(self.logits, self.label_batch)
        self.loss_dice=dice_coef_loss(self.prob,self.label_batch)
        self.dice=1-loss_dice
        self.all_trainable =tf.trainable_variables()
        self.restore_var = tf.global_variables()
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimiser = tf.train.MomentumOptimizer(self.learning_rate,0.99)
        self.gradients = tf.gradients(self.loss_dice, self.all_trainable)
        self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,1.)
        self.train_op = optimiser.apply_gradients(zip(self.clipped_gradients, self.all_trainable))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(var_list=self.restore_var, max_to_keep=40)
        if self.restore_from is not None:
            loader = tf.train.Saver(var_list=self.restore_var)
            load(loader, self.sess, self.restore_from,self.model_dir)
        
    def do_train(self):
        pre_loss=1
        dataset=LungDataset( self.img_dir,augument=True)
        lr=0.01
        for i in range(self.max_run):
            self.dice_loss_meter.reset()
            start_time = time.time()
            train_loader = DataLoader(dataset,batch_size = self.batch_size,shuffle = True,num_workers = 1,pin_memory=True,drop_last=True)
            for batch_idx, (img_, mask_,file) in enumerate(train_loader):
                self.global_steps+=1
                img=img_.numpy()
                mask=mask_.numpy()
                img=img.transpose([0,2,3,4,1])
                mask=mask.transpose([0,2,3,4,1])
                feed_dict={self.image_batch:img,self.label_batch:mask,self.learning_rate:lr}
                _,prob_=self.sess.run([self.train_op,self.prob],feed_dict=feed_dict)
                # summary_writer.add_summary(summary, iters)
                dice=self.loss_dice.eval(feed_dict,session=self.sess)
                dice_loss_meter.add(dice)
                if batch_idx>10:
                    vis.plot('dice_loss',self.dice_loss_meter.value()[0])
                    vis.plot("dice",1-self.dice_loss_meter.value()[0])
                img_k=np.zeros((64*8,64*8),dtype=np.float32)
                mask_k=np.zeros((64*8,64*8),dtype=np.float32)
                pred_k=np.zeros((64*8,64*8),dtype=np.float32)
                l=0
                for i_ in range(8):
                for j in range(8):
                    img_k[i_*64:i_*64+64,j*64:j*64+64]=img[0,l,:,:,0]
                    mask_k[i_*64:i_*64+64,j*64:j*64+64]=mask[0,l,:,:,0]
                    pred_k[i_*64:i_*64+64,j*64:j*64+64]=prob_[0,l,:,:,0]
                    l=l+1
                if np.sum(prob_)<5:
                    vis.plot('pred__',np.sum(prob_))
                vis.img('input',torch.from_numpy(img_k))
                vis.img('mask',torch.from_numpy(mask_k))
                vis.img('pred',torch.from_numpy(pred_k))
                
                if self.global_steps%50==0:
                    logitss=self.logits.eval(feed_dict,session=self.sess)
                    print("logits  %.4f"%np.sum(logitss))
                    losss=self.cross_loss.eval(feed_dict,session=self.sess)
                    dice=self.loss_dice.eval(feed_dict,session=self.sess)
                    print("Epoch: [%2d]  [%4d] ,time: %4.4f,dice_loss:%.8f,dice:%.8f,cross_loss:%.8f" % \
                          (i,batch_idx,time.time() - start_time,dice,1-dice,losss))
        if self.dice_loss_meter.value()[0]>pre_loss:
            lr=lr*0.95
            print "pre_loss: ",pre_loss," now_loss: ",self.dice_loss_meter.value()[0]," lr: ",lr
        pre_loss = self.dice_loss_meter.value()[0]
        if lr<1e-7:
            save(self.saver,self.sess,self.restore_from,self.global_steps,self.model_dir,train_tag="mask_predict")
            print "stop for lr<1e-7"
            break
        if i%10==0:
            save(self.saver,self.sess,self.restore_from,self.global_steps,self.model_dir,train_tag="mask_predict")

            
if __main__='__main__':
    restore_from='./models'
    model_dir="unet3d_v1"
    logs_dir='./log/unet3d'
    img_dir="/home/x/dcsb/data/TianChi/train/"
    trainer=Trainer(restore_from,model_dir,logs_dir,img_dir)
    
        
        
        