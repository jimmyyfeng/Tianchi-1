#encoding:utf-8
import os
import time
from tqdm import tqdm
from glob import glob
import numpy as np
import caffe
import random
from data.util import normalize
def test_cls_single_model(model_def,model_weight,data_path,save_dir=None,batch_size=8,topN=20):
    '''
    对单个分类模型进行测试
    model_def:模型定义prototxt文件
    model_weight：训练所得的模型权重文件
    data_path:数据保存父路径，路径下保存的是*.npy文件,包括节点文件和中心保存文件
    '''
    sample_list=open("/home/x/dcsb/Tianchi_caffe/train_cls.txt", 'r').read().splitlines()
    nodule_list=[r[:-2] for r in sample_list ]
    random.shuffle(nodule_list)
    net=caffe.Net(model_def,model_weight,caffe.TEST)
    if save_dir is not None:
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)
    print nodule_list[:10]
    print data_path
    for i in range(len(nodule_list)):
        if os.path.exists('/tmp/dcsb'):
            import ipdb
            ipdb.set_trace()
        patient_name=nodule_list[i].split('/')[-1]
        img=normalize(np.load(nodule_list[i]))
        if img.shape[0]<40:
            img_dimZ=img.shape[0]
            img_pad=np.zeros(40,40,40)
            center=np.array(img.shape)/2
            img_pad[20-img_dimZ/2:20+img_dimZ-img_dimZ/2,:,:]=img[:,center[1]-20:center[1]+20,center[2]-20:center[2]+20]
            net.blobs['data'].data[...]=img_pad[np.newaxis,np.newaxis,:,:,:]
            result = net.forward()['probs'][0]
            np.save(save_dir+patient_name,result)
        else:
            center=np.array(img.shape)/2
            net.blobs['data'].data[...]=img[center[0]-20:center[0]+20,center[1]-20:center[1]+20,center[2]-20:center[2]+20]
            result = net.forward()['probs'][0]
            np.save(save_dir+patient_name,result)
        if i%20==0:
            print i," hava done" 
model_def="prototxt/cls_res_val.prototxt"
model_weight="snashots/cls_res_0912_iter_2816.caffemodel"
data_path="/mnt/7/train_nodule_candidate/"
test_cls_single_model(model_def,model_weight,data_path,save_dir='/mnt/7/train_prob_cls_res_val/')

