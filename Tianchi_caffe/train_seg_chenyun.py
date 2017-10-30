#encoding:utf-8
import numpy as np

from data.data.dataset import ClsDataset,Dataset3 as Dataset
#import torch as t
#global t
#import torchnet as tnt
#from utils.visualize import Visualizer
#vis = Visualizer("simple_seg")
import os
import sys
from config import opt
from data.data.dataloader import DataLoader
if opt.online:
    os.environ["PYTHONPATH"]="%s:%s"%("/home/caffe/python","/workspace/pai")
    sys.path.append("/home/caffe/python")
    sys.path.append("/workspace/pai")
#import ipdb;ipdb.set_trace()
from config import opt
import time
import caffe
import logging
import psutil
from utils.confuse import ConfusionMeter
#from utils import sysinfo
import subprocess
from tqdm import tqdm


logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='myapp.log', #修改log地址为跳板机上的文件
                filemode='w')




# 数据读取4核 dataloader.num_workers=4，下标：64-67
# 运行程序是60核，下标：4-63
# PAI平台占用 下标：0-3
if opt.online==False:
    os.environ["OMP_NUM_THREADS"]='55'
    os.environ["KMP_AFFINITY"]="explicit,1,0,granularity=fine,proclist=[4-58]"
else:
    os.environ["OMP_NUM_THREADS"]='60'
    os.environ["KMP_AFFINITY"]="explicit,1,0,granularity=fine,proclist=[4-63]"

seg_model_path=None#'snashots/simple_seg_iter_6000.caffemodel'#opt.model_seg_pre_weight
def train_seg():
    max_epoch=100
    #sample_list=open("/home/x/data/dataset/tianchi/train.txt", 'r').read().splitlines()
    # dataset=Dataset('/mnt/7/train_nodule_cubic/',augment=True)
    # dataloader = DataLoader(dataset,batch_size=4,       
    #                     num_workers=1,
    #                     shuffle=True,
    #                     drop_last=True
    #                     )
    solver = caffe.AdamSolver('solver/seg_chenyun.prototxt')
    if seg_model_path is not None:
        print "loading pre-model.",seg_model_path
        #solver.restore(seg_model_path)
        solver.net.copy_from(seg_model_path)
    confusem = ConfusionMeter(2)
    _train_loss=0
    #loss_meter = tnt.meter.AverageValueMeter()
    #loss_meter.reset()
    for it in range(max_epoch):
        confusem.reset()
        _accuracy=0
        display=120
        for ii  in tqdm(range(10)):
            if os.path.exists("/tmp/debug"):
                import ipdb; ipdb.set_trace()
            data_ = np.random.randn(4,1,64,64,64) 
            solver.net.blobs["data"].data[...]= data_
            solver.net.blobs["label"].data[...]=np.zeros_like(data_) 
            solver.step(1) 
            now_loss=solver.net.blobs["loss"].data 
            # 每迭代一次，训练batch_size张图片  
            _train_loss += now_loss
            if ii % 120 == 0:  
                # 计算平均train loss  
                train_loss = _train_loss / 120         
                print "step_display : %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss
                _train_loss = 0  

    del solver  

# 
train_seg()