import argparse
from datetime import datetime
import os
import sys
import time
import random
from Classifier_3d_v1 import Classifier
import tensorflow as tf
from util import Visualizer
import numpy as np
from dataset_classifier import LungDataset
import torch
from ops import load,save,pixelwise_cross_entropy
import torchnet as tnt
from torch.utils.data import DataLoader
#restore_from='./models'
restore_from=None
models_path='./models'
logs='./logs'
luna="/home/x/dcsb/data/TianChi/"
luna_data="/home/x/data/datasets/tianchi/train/"
batch_size = 1
max_run    = 1000
epoch_print  = 100
iters=0 
vis = Visualizer()


def main():
    vis.vis.texts=''
    dice_loss_meter =tnt.meter.AverageValueMeter()
    image_batch=tf.placeholder(tf.float32, shape=[None, 48, 48,48, 1])
    label_batch=tf.placeholder(tf.float32, shape=[None,2])
    net=Classifier({'data': image_batch},batch_size=batch_size)
    prob = net.layers['result']
    logits=net.layers['logits']
    dataset=LungDataset("/home/x/dcsb/data/TianChi",augument=True)
    
    all_trainable =tf.trainable_variables()
    restore_var = tf.global_variables()
    
    cross_loss = tf.losses.softmax_cross_entropy(label_batch,logits)
    
    global iters
    cross_loss_sum=tf.summary.scalar("crossloss",cross_loss)
    # accuracy=tf.metrics.accuracy(label_batch,prob)
    optimiser = tf.train.MomentumOptimizer(0.01,0.99)
    gradients = tf.gradients(cross_loss, all_trainable)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,1.)
    train_op = optimiser.apply_gradients(zip(clipped_gradients, all_trainable))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    all_sum=tf.summary.merge([cross_loss_sum])
    summary_writer = tf.summary.FileWriter(logs,graph=tf.get_default_graph())
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=40)
    
    # Load variables if the checkpoint is provided.
    if restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, restore_from,"classifier_v2")
    
    for i in range(max_run):
        dice_loss_meter.reset()
        start_time = time.time()
        labels=np.array([1,0])
        labels=labels[np.newaxis,:]
        pred=np.array([1,0])
        pred=pred[np.newaxis,:]
        train_loader = DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = 1,pin_memory=True,drop_last=True)
        for batch_idx, (img_, label_,_) in enumerate(train_loader):
            iters+=1
            img=img_.numpy()
            label=label_.numpy()
            labels=np.concatenate([labels,label],axis=0)
            img=img.transpose([0,2,3,4,1])
            feed_dict={image_batch:img,label_batch:label}
            _,cross_loss_,probs,summary=sess.run([train_op,cross_loss,prob,all_sum],feed_dict=feed_dict)
            summary_writer.add_summary(summary, iters)
            pred=np.concatenate([pred,probs],axis=0)
            # print "prob+:",probs[:,0]
            vis.plot('accuracy',np.mean(np.argmax(labels,axis=1)==np.argmax(pred,axis=1)))
            dice_loss_meter.add(cross_loss_)
            if batch_idx>10:
                try:
                    vis.plot('cross_loss',dice_loss_meter.value()[0])
                except:
                    pass
            vis.img('input',img_[0,0,24,:,:].cpu().float())
            if iters%50==0:
                
                pred_=np.argmax(pred,axis=1)
                label_=np.argmax(labels,axis=1)
                acc=np.mean(label_==pred_)
                cross=cross_loss.eval(feed_dict,session=sess)
                print("Epoch: [%2d]  [%4d] ,time: %4.4f,cross_loss:%.8f,accuracy:%.8f"% \
                      (i,batch_idx,time.time() - start_time,cross,acc))
            
        if i%2==0:
            save(saver,sess,models_path,iters,"classifier_v2",train_tag="nodule_predict")
main()
    
