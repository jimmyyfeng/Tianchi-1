import numpy as np
import os
from unet3D import Unet3D
import tensorflow as tf
from ops import load
from cysb import get_ct,cropBlocks,normalize
data='/home/x/dcsb/data/TianChi/'
restore_from='./models'
luna="/home/x/data/datasets/tianchi/"
batch_size=1
image_batch=tf.placeholder(tf.float32, shape=[None, None, None,None, 1])
def do_seg(file_name,sess,net):
    
    prob = net.layers['result']
    print "preparing data................................."
    img_arr,origin,spacing=get_ct(luna+'validation/'+file_name+'.mhd')
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
        feed_dict={image_batch:blocks[i][np.newaxis,:,:,:,np.newaxis]} 
        probs[off_min[0]+indexs[i,0]*64:off_min[0]+indexs[i,0]*64+64,
            off_min[1]+indexs[i,1]*64:off_min[1]+indexs[i,1]*64+64,
            off_min[2]+indexs[i,2]*64:off_min[2]+indexs[i,2]*64+64,
             ]=prob.eval(feed_dict,session=sess)[0,:,:,:,0]
        if i%50==0:
            print "doing with:",i
    np.save(data+file_name+".npy",probs)
    
def CreateSegNet():
    net=Unet3D({'data': image_batch},batch_size=batch_size,is_training=False)        
    restore_var = tf.global_variables()
    
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
  
    if restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, restore_from,"unet3d")
    return net,sess

file_name='LKDS-00071'
net,sess=CreateSegNet()
do_seg(file_name,sess,net)