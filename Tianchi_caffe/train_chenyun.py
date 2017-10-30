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
try:
    from tqdm import tqdm
except:
    tqdm = lambda x:x
#from utils import sysinfo
import subprocess
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
cls_model_path=None
def train_chenyun_cls():
    max_epoch=100
    sample_list=open("/home/x/dcsb/Tianchi_caffe/train_cls.txt", 'r').read().splitlines()
    nodule_list=[r[:-2] for r in sample_list if r[-1]=='1']
    background_list=[r[:-2] for r in sample_list if r[-1]=='0']
    dataset=ClsDataset(nodule_list,background_list,ratio=2,augument=True,phase="train",crop_size=[16,16,16])
    dataloader = DataLoader(dataset,batch_size=32,       
                        num_workers=1,
                        shuffle=True,
                        drop_last=True
                        )
    solver = caffe.AdamSolver('solver/chenyun.prototxt')
    if cls_model_path is not None:
        print "loading pre-model.",cls_model_path
        solver.restore(cls_model_path)
        #solver.net.copy_from(cls_model_path)
    confusem = ConfusionMeter(2)

    for it in range(max_epoch):
        if os.path.exists("/tmp/debug"):
            import ipdb; ipdb.set_trace()
        confusem.reset()
        _train_loss=0

        _accuracy=0
        display=120
        for ii, (input, label) in tqdm(enumerate(dataloader)):
            solver.net.blobs["data"].data[...]=input[:,np.newaxis,:,:,:]
            solver.net.blobs["label"].data[...]=label
            solver.step(1) 
            
            now_loss=solver.net.blobs["loss"].data 
            score=solver.net.blobs["cls"].data 
            target=label
            confusem.add(score, target)
            _train_loss += now_loss

            if ii%5==0:
                print "------------cm---------------"
                print('cm:%s' % (str(confusem.value)))
              
            if ii % 120 == 0:  
                # 计算平均train loss  
                train_loss = _train_loss / 120             
                print "step_display : %s train_loss_1 average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss
                _train_loss= 0  
    del solver  

train_chenyun_cls()
