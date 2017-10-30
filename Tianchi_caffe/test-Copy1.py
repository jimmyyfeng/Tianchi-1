#encoding:utf-8
import os
import time
from tqdm import tqdm
import numpy as np
from skimage import color, data, measure, morphology, segmentation
from data.util import check_center,cropBlocks,load_ct,normalize
import caffe
from config import opt
def seg(net,file_name,seg_size,img_new):
    '''
    用CPU跑特别慢 确实GPU很有必要
    '''
    print " data prepared................................."
    depth, height, width = img_new.shape
    #off_min =np.array([0,0,0])
    num = np.array(img_new.shape) / seg_size
    off = np.array(img_new.shape) - seg_size * num
    off_min = off / 2
    blocks, indexs = cropBlocks(img_new,seg_size,off_min)
    probs1 = np.zeros(img_new.shape, dtype=np.float32)
    batch_num=opt.batch_size_test_seg
    print "doing on patient:", file_name

    for i in range(blocks.shape[0]):
        if (i % batch_num == batch_num - 1):
            batch_inputs_numpy = [blocks[j][np.newaxis, np.newaxis, :, :, :] for j in range(i - batch_num + 1, i + 1)]
            net.blobs['data'].data[...]=batch_inputs_numpy
            batch_outputs = net.forward()
            for j in range(i - batch_num + 1, i + 1):
                probs1[off_min[0] + indexs[j, 0] * seg_size[0]:off_min[0] + indexs[j, 0] * seg_size[0] + seg_size[0],
                      off_min[1] + indexs[j, 1] * seg_size[1]:off_min[1] + indexs[j, 1] * seg_size[1] + seg_size[1],
                      off_min[2] + indexs[j, 2] * seg_size[2]:off_min[2] + indexs[j, 2] * seg_size[2] + seg_size[2],
                      ] = batch_outputs['probs'][0,0]
        if i%50==0:
            print i," have finished"
    print "probs1.max()",probs1.max()

    return probs1



    
def test_seg(save_dir,data_path,model_def,model_weight):
    is_save=True
    prob_threshould=0.4#二值化阈值
    save_dir=opt.save_dir#切块保存路径
    data_list=indices = open(data_path, 'r').read().splitlines()
    start=time.time()
    net=caffe.Net(model_def,model_weight,caffe.TEST)
    for file_name in tqdm(data_list[:2]):
        img_arr,origin,spacing=load_ct(file_name)
        img_new=normalize(img_arr)
        seg_size = [64,64,64] 
        crop_z=64
        dim_z=img_arr.shape[0]
        if dim_z<seg_size[0]:
            crop_z=img_arr.shape[0]
            img_pad=np.zeros(img_arr.shape,dtype=np.float32)
            img_pad[32-dim_z/2:32+dim_z-dim_z/2,:,:]=img_new
            probs1=seg(net2,file_name,seg_size,img_pad)
            probs1=probs1[32-dim_z/2:32+dim_z-dim_z/2]
            del img_pad
        else:
            probs1=seg(net2,file_name,seg_size,img_new)
        
        
        seg_size = [80,80,80] 
        dim_z=img_arr.shape[0]
        if dim_z<seg_size[0]:
            img_pad=np.zeros(img_arr.shape,dtype=np.float32)
            img_pad[40-dim_z/2:40+dim_z-dim_z/2,:,:]=img_new
            probs2=seg(net1,file_name,seg_size,img_pad)
            probs2=probs2[40-dim_z/2:40+dim_z-dim_z/2]
            del img_pad
        else:
            probs2=seg(net1,file_name,seg_size,img_new)
            #seg_size=[img_arr.shape[0],80,80]
        #net.blobs['data'].reshape(opt.batch_size,1,seg_size[0],seg_size[1],seg_size[2])
        #import ipdb; ipdb.set_trace()
        probs=(probs1+probs2)/2.0
        np.save(mhd_name+"_probs.npy",probs)
        crop_size=[crop_z,64,64]
        mhd_name=file_name.split('/')[-1][:-4]
        probs=probs>prob_threshould 
        probs=morphology.dilation(probs,np.ones([3,3,3]))
        probs=morphology.dilation(probs,np.ones([3,3,3]))
        probs=morphology.erosion(probs,np.ones([3,3,3]))
        np.save(file_name+"_probs.npy",probs)
        labels = measure.label(probs,connectivity=2)
        #label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        centers = []
        crops = []
        bboxes = []
        spans=[]
        for prop in regions:
            B = prop.bbox
            if B[3]-B[0]>2 and B[4]-B[1]>4 and B[5]-B[2]>4:
                z=int((B[3]+B[0])/2.0)
                y=int((B[4]+B[1])/2.0)
                x=int((B[5]+B[2])/2.0)
                span=np.array([int(B[3]-B[0]),int(B[4]-B[1]),int(B[5]-B[2])])
                spans.append(span)
                centers.append(np.array([z,y,x]))
                bboxes.append(B)
        for idx,bbox in enumerate(bboxes):
            crop=np.zeros(crop_size,dtype=np.float32)
            crop_center=centers[idx]
            half=np.array(crop_size)/2
            crop_center=check_center(crop_size,crop_center,img_arr.shape)
            crop=img_arr[int(crop_center[0]-half[0]):int(crop_center[0]+half[0]),\
                         int(crop_center[1]-half[1]):int(crop_center[1]+half[1]),\
                         int(crop_center[2]-half[1]):int(crop_center[2]+half[1])]
            crops.append(crop)
        if is_save:
            np.save(save_dir+mhd_name+"_nodule.npy",np.array(crops))
            np.save(save_dir+mhd_name+"_center.npy",np.array(centers))
            np.save(save_dir+mhd_name+"_size.npy",np.array(spans))
        
save_dir='/mnt/7/0827_caffe_80/'#切块保存路径
data_path="/home/x/data/datasets/tianchi/txtfiles/train.txt"
model_def="prototxt/simple_seg_val.prototxt"
model_weight="snashots/simple_seg_iter_1840.caffemodel"
test_seg(save_dir,data_path,model_def,model_weight)    
    
