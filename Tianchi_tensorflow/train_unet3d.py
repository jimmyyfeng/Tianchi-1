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
restore_from='./models'
#restore_from=None
logs='./log/unet3d'
models_path='./models'
luna="/home/x/dcsb/data/TianChi/"
luna_data="/home/x/data/datasets/tianchi/"
batch_size = 2
max_run    = 1000
epoch_print  = 100
iters=0 
vis = Visualizer()




def main():
    pre_loss = 1
    dice_loss_meter =tnt.meter.AverageValueMeter()
    vis.vis.texts=''
    # loss_meter =tnt.meter.AverageValueMeter()
    image_batch=tf.placeholder(tf.float32, shape=[None, 64, 64,64, 1])
    label_batch=tf.placeholder(tf.float32, shape=[None, 64, 64,64, 1])
    net=Unet3D({'data': image_batch},batch_size=batch_size,keep_prob=0.5)
    dataset=LungDataset(luna_data+'train/',augument=True)
    prob = net.layers['result']
    logits=net.layers['conv_8']
    logitsum=tf.summary.histogram("logits",logits)
    conv7_1=net.layers['conv7_1']
    conv7_1_sum=tf.summary.histogram("conv7_1",conv7_1)
    conv7_2=net.layers['conv7_2']
    con7_2sum=tf.summary.histogram("conv7_2",conv7_2)
    print "logits--------------:",logits.shape
    cross_loss = pixelwise_cross_entropy(logits, label_batch)
    # cross_loss_sum=tf.summary.scalar("cross_loss",cross_loss)
    all_trainable =tf.trainable_variables()
    restore_var = tf.global_variables()
    
    loss_dice=dice_coef_loss(prob,label_batch)
    dice=1-loss_dice
    
    extra_loss=extraLoss(prob,label_batch)
    Loss=loss_dice#-0.001*tf.norm(prob-0.5)#+10*extraLoss1(prob,label_batch)
    # dice_sum=tf.summary.scalar("dice",dice)
    global iters
    learning_rate = tf.placeholder(tf.float32)
    # lr_sum=tf.summary.scalar("learning_rate",learning_rate)
    optimiser = tf.train.MomentumOptimizer(learning_rate,0.99)
    gradients = tf.gradients(Loss, all_trainable)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,1.)
    train_op = optimiser.apply_gradients(zip(clipped_gradients, all_trainable))
    
    # summarypic=prob[:,32]
    # origin_sum=tf.summary.image("image_batch",image_batch[:,32,:,:])
    # mask_sum=tf.summary.image("label_batch",label_batch[:,32,:,:]+image_batch[:,32,:,:])
    # img_sum=tf.summary.image("prediction",tf.add(summarypic,image_batch[:,32,:,:]))
    # summary_writer = tf.summary.FileWriter(logs,graph=tf.get_default_graph())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    # all_sum=tf.summary.merge([cross_loss_sum,img_sum,origin_sum,mask_sum,dice_sum,logitsum,conv7_1_sum,con7_2sum])
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=40)
    lr=0.01
    # Load variables if the checkpoint is provided.
    if restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, restore_from,"unet3d_v1")
    for i in range(max_run):
        dice_loss_meter.reset()
        start_time = time.time()
        train_loader = DataLoader(dataset,batch_size = batch_size,shuffle = True,num_workers = 1,pin_memory=True,drop_last=True)
        for batch_idx, (img_, mask_,file) in enumerate(train_loader):
            iters+=1
            img=img_.numpy()
            mask=mask_.numpy()
            img=img.transpose([0,2,3,4,1])
            mask=mask.transpose([0,2,3,4,1])
            feed_dict={image_batch:img,label_batch:mask,learning_rate:lr}
            _,prob_=sess.run([train_op,prob],feed_dict=feed_dict)
            # summary_writer.add_summary(summary, iters)
            dice=loss_dice.eval(feed_dict,session=sess)
            dice_loss_meter.add(dice)
            all_loss=Loss.eval(feed_dict,session=sess)
            if batch_idx>10:
                vis.plot('dice_loss',dice_loss_meter.value()[0])
                vis.plot("dice",1-dice_loss_meter.value()[0])
            # vis.plot("all_loss",all_loss)
            # vis.img('input',img_[0,0,32,:,:].cpu().float())
            # vis.img('mask',mask_[0,0,32,:,:].cpu().float())
            
            img_k=np.zeros((64*8,64*8),dtype=np.float32)
            mask_k=np.zeros((64*8,64*8),dtype=np.float32)
            pred_k=np.zeros((64*8,64*8),dtype=np.float32)
            l=0
            # print file
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
            # if dice<0.01:
#                 img_l=np.zeros((64*5,64*5),dtype=np.float32)
#                 mask_l=np.zeros((64*5,64*5),dtype=np.float32)
#                 pred_l=np.zeros((64*5,64*5),dtype=np.float32)
#                 l=0
#                 for i in range(5):
#                     for j in range(5):
#                         l=l+1
#                         img_l[i*64:i*64+64,j*64:j*64+64]=img[0,12+l,:,:,0]
#                         mask_l[i*64:i*64+64,j*64:j*64+64]=mask[0,12+l,:,:,0]
#                         pred_l[i*64:i*64+64,j*64:j*64+64]=prob_[0,12+l,:,:,0]
                
#                 vis.img('input_0.01_loss',torch.from_numpy(img_l))
#                 vis.img('mask_0.01_loss',torch.from_numpy(mask_l))
#                 vis.img('pred_0.01_loss',torch.from_numpy(pred_l))
                
            if iters%50==0:
                logitss=logits.eval(feed_dict,session=sess)
                print("logits  %.4f"%np.sum(logitss))
                losss=cross_loss.eval(feed_dict,session=sess)
                dice=loss_dice.eval(feed_dict,session=sess)
                all_loss=Loss.eval(feed_dict,session=sess)
                print("Epoch: [%2d]  [%4d] ,time: %4.4f,all_loss:%.8f,dice_loss:%.8f,dice:%.8f,cross_loss:%.8f" % \
                      (i,batch_idx,time.time() - start_time,all_loss,dice,1-dice,losss))
        if dice_loss_meter.value()[0]>pre_loss:
            lr=lr*0.95
            print "pre_loss: ",pre_loss," now_loss: ",dice_loss_meter.value()[0]," lr: ",lr
        pre_loss = dice_loss_meter.value()[0]
        if lr<1e-7:
            save(saver,sess,models_path,iters,"unet3d_v1",train_tag="mask_predict")
            print "stop for lr<1e-7"
            break
        if i%10==0:
            save(saver,sess,models_path,iters,"unet3d_v1",train_tag="mask_predict")
main()    