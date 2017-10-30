#encoding:utf-8
import caffe
from glob import glob
import numpy as np
from tqdm import tqdm
import random
model_def="prototxt/cls_multi_kernel_val.prototxt"
model_weight="snashots/cls_multi_kernel_0905_iter_1920.caffemodel"

def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    '''
    @image：原始数据
    @MIN_BOUND：最小值
    @MAX_BOUND：最大值
    Return：归一化后的image
    ！TODO：数据截断归一化
    '''
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image  
net=caffe.Net(model_def,model_weight,caffe.TEST)
sample_list=open("/home/x/dcsb/Tianchi_caffe/val_cls.txt", 'r').read().splitlines()
nodule_list=[r[:-2] for r in sample_list if r[-1]=='1']
background_list=[r[:-2] for r in sample_list if r[-1]=='0']
random.shuffle(background_list)
random.shuffle(nodule_list)
background_list=background_list[:3000]
nodule_bum=len(nodule_list)
backgd_num=len(background_list)
print "pos samples num: ",nodule_bum
print "neg samples num: ",backgd_num


import os
from utils.confuse import ConfusionMeter
confusem=ConfusionMeter(2)
if True:
    sample_list=nodule_list+background_list
    labels_list=np.array([1]*nodule_bum+[0]*backgd_num)

    index=np.arange(0,len(sample_list))
    import random
    random.shuffle(index)
    result=np.zeros(len(sample_list))
    labels_=labels_list[index]
    batch_size=8
    for ii in tqdm(range(len(sample_list))):
        if os.path.exists('/tmp/debug'):
            import ipdb;ipdb.set_trace()
        if ii%batch_size==7:
            img=np.array([normalize(np.load(sample_list[index[rr]])) for rr in range(ii-batch_size+1,ii+1)])
            net.blobs["data"].data[...]=img[:,np.newaxis,32-20:32+20,32-20:32+20,32-20:32+20]
            output=net.forward()["probs"]
            result[ii-batch_size+1:ii+1]=output.argmax(axis=1)
            confusem.add(output,labels_[ii-batch_size+1:ii+1])
            print confusem.value
    print "acc:", np.sum(result==labels_)*1.0/len(sample_list)
