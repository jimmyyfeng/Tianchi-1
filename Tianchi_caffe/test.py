#encoding:utf-8
import os
import time
from tqdm import tqdm
from glob import glob
import numpy as np
from skimage import color, data, measure, morphology, segmentation
from data.util import check_center,cropBlocks,load_ct,normalize,get_topn,voxel_2_world
import caffe
from config import opt
import csv
import random
def seg(net,file_name):
    '''
    用CPU跑特别慢 确实GPU很有必要
    '''
    print " data prepared................................."
    img_arr,origin,spacing=load_ct(file_name)
    seg_size = [80,80,80] 
    #if img_arr.shape[0]<seg_size[0]:
    #    seg_size=[img_arr.shape[0],80,80]
    #net.blobs['data'].reshape(opt.batch_size,1,seg_size[0],seg_size[1],seg_size[2])
    img_new=normalize(img_arr)
    depth, height, width = img_new.shape
    off_min =np.array([0,0,0])
    #num = np.array(img_new.shape) / seg_size
    #off = np.array(img_new.shape) - seg_size * num
    #off_min = off / 2
    blocks, indexs = cropBlocks(img_new,seg_size,off_min)
    probs1 = np.zeros(img_new.shape, dtype=np.float32)
    batch_num=2
    print "doing on patient:", file_name

    for i in range(blocks.shape[0]):
        if (i % batch_num == batch_num - 1):
            batch_inputs_numpy = [blocks[j][np.newaxis, np.newaxis, :, :, :] for j in range(i - batch_num + 1, i + 1)]
            #import ipdb;ipdb.set_trace()
            net.blobs['data'].data[...]=batch_inputs_numpy
            #print net.blobs['data'].data.shape
            batch_outputs = net.forward()
            for j in range(i - batch_num + 1, i + 1):
                probs1[off_min[0] + indexs[j, 0] * seg_size[0]:off_min[0] + indexs[j, 0] * seg_size[0] + seg_size[0],
                      off_min[1] + indexs[j, 1] * seg_size[1]:off_min[1] + indexs[j, 1] * seg_size[1] + seg_size[1],
                      off_min[2] + indexs[j, 2] * seg_size[2]:off_min[2] + indexs[j, 2] * seg_size[2] + seg_size[2],
                      ] = batch_outputs['probs'][j - (i - batch_num + 1),0]
        if i%50==0:
            print i," have finished"
    print "probs1.max()",probs1.max()
    #seg_size = [64,64,64] 
    #if img_arr.shape[0]<seg_size[0]:
    #    seg_size=[img_arr.shape[0],64,64]
    #net.blobs['data'].reshape(opt.batch_size,1,seg_size[0],seg_size[1],seg_size[2])
    off_min=np.array([40,40,40])
    #num = np.array(img_new.shape) / seg_size
    #off = np.array(img_new.shape) - seg_size * num
    #off_min = off / 2
    blocks, indexs = cropBlocks(img_new,seg_size,off_min)
    probs2 = np.zeros(img_new.shape, dtype=np.float32)
    batch_num=2
    print "doing on patient:", file_name

    for i in range(blocks.shape[0]):
        if (i % batch_num == batch_num - 1):
            batch_inputs_numpy = [blocks[j][np.newaxis, np.newaxis, :, :, :] for j in range(i - batch_num + 1, i + 1)]
            net.blobs['data'].data[...]=batch_inputs_numpy
            batch_outputs = net.forward()
            for j in range(i - batch_num + 1, i + 1):
                probs2[off_min[0] + indexs[j, 0] * seg_size[0]:off_min[0] + indexs[j, 0] * seg_size[0] + seg_size[0],
                      off_min[1] + indexs[j, 1] * seg_size[1]:off_min[1] + indexs[j, 1] * seg_size[1] + seg_size[1],
                      off_min[2] + indexs[j, 2] * seg_size[2]:off_min[2] + indexs[j, 2] * seg_size[2] + seg_size[2],
                      ] =batch_outputs['probs'][j - (i - batch_num + 1),0]
        if i%50==0:
            print i," have finished"
    print "probs2.max()",probs2.max()
    return (probs1+probs2)/2.0,img_arr



    
def test_seg(save_dir,data_path,model_def,model_weight):
    is_save=True
    crop_size=[64,64,64]
    prob_threshould=0.4#二值化阈值
    data_list=indices = open(data_path, 'r').read().splitlines()
    #data_list=["/home/x/data/datasets/tianchi/train/LKDS-00004.mhd"]
    data_list.sort()
    start=time.time()
    net=caffe.Net(model_def,model_weight,caffe.TEST)
    print "time:",time.time()-start
    for file_name in tqdm(data_list):
        mhd_name=file_name.split('/')[-1][:-4]
        if os.path.exists(save_dir+mhd_name+"_nodule.npy"):
            print mhd_name ," have finished"
            continue
        probs,img_arr=seg(net,file_name)
        np.save(mhd_name+"_probs.npy",probs)
        probs=probs>prob_threshould 
        probs=morphology.dilation(probs,np.ones([3,3,3]))
        probs=morphology.dilation(probs,np.ones([3,3,3]))
        probs=morphology.erosion(probs,np.ones([3,3,3]))
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
def write_csv(world,probability,csv_writer,patient_id,threshold=0.):
    '''
    @world:世界坐标，numpy （N，3）N为结点数目，坐标排序为X,Y,Z
    @probability：概率值，numpy（N,2），第一列为正概率
    @csv_writer：csv文件读写器
    @threshold：概率阈值，大于此阈值的概率才写到csv文件
    Return：None
    TODO：将样本的分类结果写入csv文件
    '''
    for j in range(world.shape[0]):
        if probability[j]>threshold:
            row=list(world[j])
            row.append(probability[j])
            row=[patient_id]+row
            # print row
            csv_writer.writerow(row)
def do_class(imgs,net,batch_size):
    '''
    @img:待送入模型的图像，numpy（N,D,D,D）,imgs已经被pad到batchsize的整数倍
    @model：用于分类的名
    Return：result，Numpy分类结果
    TODO：对检测到的结点进行二值分类
    '''
    
    length=imgs.shape[0]    
    result=np.zeros([length])
    print "length: ",length
    for i in range(length):
        if i%batch_size==batch_size-1:
            batch_inputs_numpy = [imgs[j][np.newaxis, np.newaxis, :, :, :] for j in range(i - batch_size + 1, i + 1)]
            net.blobs['data'].data[...]=batch_inputs_numpy
            result[i - batch_size + 1:i + 1] = net.forward()['probs'][:,1]
    return result            
def test_cls_single_model(model_def,model_weight,data_path,batch_size=8,topN=20):
    '''
    对单个分类模型进行测试
    model_def:模型定义prototxt文件
    model_weight：训练所得的模型权重文件
    data_path:数据保存父路径，路径下保存的是*.npy文件,包括节点文件和中心保存文件
    '''
    nodule_list=glob(data_path+"*_nodule.npy")
    random.shuffle(nodule_list)
    center_list=glob(data_path+"*_center.npy")
    net=caffe.Net(model_def,model_weight,caffe.TEST)
    print nodule_list[:10]
    print data_path
    f=open("tmp.csv", "wa")
    csv_writer = csv.writer(f, dialect="excel")
    csv_writer.writerow(
        ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
    for i,patient in enumerate(nodule_list):
        if os.path.exists('/tmp/dcsb'):
            import ipdb
            ipdb.set_trace()
        patient_id=patient.split('/')[-1].split('_')[-2]
        patient_nodule=normalize(np.load(patient))#导入结点文件
        nodule_centers=np.load(patient.replace("nodule","center"))
        length=patient_nodule.shape[0]
        if length%batch_size!=0:
            z_img=length/batch_size
            z_pad=(z_img+1)*batch_size
            img_pad=np.zeros([z_pad,patient_nodule.shape[1],patient_nodule.shape[2],patient_nodule.shape[3]])
            img_pad[z_pad/2-length/2:z_pad/2+length-length/2]=patient_nodule
            result=do_class(img_pad[:,24-20:24+20,24-20:24+20,24-20:24+20],net,batch_size)
            result=result[z_pad/2-length/2:z_pad/2+length-length/2]
        else:result=do_class(patient_nodule[:,24-20:24+20,24-20:24+20,24-20:24+20],net,batch_size)
        if length<topN:
            topN=length
        else:
            topN=topN
        index=get_topn(result,topN)
        probability=result[index]
        center_=nodule_centers[index]
        world=voxel_2_world(center_[:,::-1],patient_id)
        write_csv(world,probability,csv_writer,patient_id)
        if i%20==0:
            print i," hava done" 
#model_def="prototxt/cls_multi_kernel_val.prototxt"
#model_weight="snashots/cls_multi_kernel_iter_2160.caffemodel"
#data_path="/mnt/7/0704_train_48_64/"
#test_cls_single_model(model_def,model_weight,data_path)


save_dir='/mnt/7/0908_caffe_train/'#切块保存路径
data_path="/home/x/data/datasets/tianchi/txtfiles/train.txt"
model_def="prototxt/simple_seg_val.prototxt"
model_weight="snashots/simple_seg_iter_1840.caffemodel"
test_seg(save_dir,data_path,model_def,model_weight)    
    
