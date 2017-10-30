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
    dataset=Dataset('/mnt/7/train_nodule/',augment=True)
    dataloader = DataLoader(dataset,batch_size=4,       
                        num_workers=1,
                        shuffle=True,
                        drop_last=True
                        )
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
        # for ii, (input, label) in enumerate(dataloader):
        import tqdm
        for i in tqdm.tqdm(range(10)):
            if os.path.exists("/tmp/debug"):
                import ipdb; ipdb.set_trace()
            # solver.net.blobs["data"].data[...]=input[:,np.newaxis,:,:,:]
            # solver.net.blobs["label"].data[...]=label[:,np.newaxis,:,:,:]

            solver.net.blobs["data"].data[...]=np.ones([4,1,64,64,64])
            solver.net.blobs["label"].data[...]=np.ones([4,1,64,64,64])

            solver.step(1) 
            now_loss=solver.net.blobs["loss"].data 
            # # 每迭代一次，训练batch_size张图片  
            # _train_loss += now_loss
            # if ii % 120 == 0:  
            #     # 计算平均train loss  
            #     train_loss = _train_loss / 120         
            #     print "step_display : %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss
            #     _train_loss = 0  

    del solver  

cls_model_path=None#"snashots/cls_multi_kernel_0905_iter_1920.solverstate"  

def train_cls():
    max_epoch=100
    sample_list=open("/home/x/dcsb/Tianchi_caffe/train_cls.txt", 'r').read().splitlines()
    nodule_list=[r[:-2] for r in sample_list if r[-1]=='1']
    background_list=[r[:-2] for r in sample_list if r[-1]=='0']
    dataset=ClsDataset(nodule_list,background_list,ratio=2,augument=True,phase="train")
    dataloader = DataLoader(dataset,batch_size=64,       
                        num_workers=1,
                        shuffle=True,
                        drop_last=True
                        )
    solver = caffe.AdamSolver('solver/cls_solver.prototxt')
    if cls_model_path is not None:
        print "loading pre-model.",cls_model_path
        solver.restore(cls_model_path)
        #solver.net.copy_from(cls_model_path)
    confusem = ConfusionMeter(2)
    for it in range(max_epoch):
        confusem.reset()
        _train_loss=0
        _accuracy=0
        display=120
        for ii, (input, label) in enumerate(dataloader):
            if os.path.exists("/tmp/debug"):
                import ipdb; ipdb.set_trace()
            solver.net.blobs["data"].data[...]=input[:,np.newaxis,:,:,:]
            solver.net.blobs["label"].data[...]=label
            solver.step(1) 
            now_loss=solver.net.blobs["loss"].data 
            score=solver.net.blobs["score"].data 
            target=label
            confusem.add(score, target)
            # 每迭代一次，训练batch_size张图片  
            _train_loss += now_loss
            _accuracy += solver.net.blobs['Accuracy1'].data
            if ii%5==0:
                print('cm:%s' % (str(confusem.value)))
            if ii % 120 == 0:  
                # 计算平均train loss  
                train_loss = _train_loss / 120         
                print "step_display : %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss
                _train_loss = 0  
            if ii%480==0:
                train_acc = _accuracy / 480  
                print "step_acc : %s train_acc average:"%(time.strftime('%m-%d %H:%M:%S')) , train_acc
                _accuracy=0
            if ii%960==959:
                confusem.reset()
   
    del solver  
#cls_model_path="snashots/cls_multi_conv_train_0909_iter_2640.solverstate"    
def train_cls_three_models():
    max_epoch=100
    sample_list=open("/home/x/dcsb/Tianchi_caffe/train_cls.txt", 'r').read().splitlines()
    nodule_list=[r[:-2] for r in sample_list if r[-1]=='1']
    background_list=[r[:-2] for r in sample_list if r[-1]=='0']
    dataset=ClsDataset(nodule_list,background_list,ratio=2,augument=True,phase="train")
    dataloader = DataLoader(dataset,batch_size=16,       
                        num_workers=1,
                        shuffle=True,
                        drop_last=True
                        )
    solver = caffe.AdamSolver('solver/cls_solver.prototxt')
    if cls_model_path is not None:
        print "loading pre-model.",cls_model_path
        solver.restore(cls_model_path)
        #solver.net.copy_from(cls_model_path)
    confusem = ConfusionMeter(2)
    confusem1 = ConfusionMeter(2)
    confusem2 = ConfusionMeter(2)
    confusem3 = ConfusionMeter(2)
    for it in range(max_epoch):
        if os.path.exists("/tmp/debug"):
            import ipdb; ipdb.set_trace()
        confusem.reset()
        _train_loss_1=0
        _train_loss_2=0
        _train_loss_3=0
        _accuracy=0
        display=120
        for ii, (input, label) in enumerate(dataloader):
            solver.net.blobs["arch1_data"].data[...]=input[:,np.newaxis,20-10:20+10,20-10:20+10,20-10:20+10]
            solver.net.blobs["arch2_data"].data[...]=input[:,np.newaxis,20-15:20+15,20-15:20+15,20-15:20+15]
            solver.net.blobs["arch3_data"].data[...]=input[:,np.newaxis,:,:,:]
            solver.net.blobs["label"].data[...]=label
            solver.step(1) 
            now_loss_1=solver.net.blobs["arch1_loss"].data 
            now_loss_2=solver.net.blobs["arch2_loss"].data 
            now_loss_3=solver.net.blobs["arch3_loss"].data 
            score_1=solver.net.blobs["arch1"].data 
            score_2=solver.net.blobs["arch2"].data 
            score_3=solver.net.blobs["arch3"].data 
            score=(score_1+score_2+score_3)/3.0
            target=label
            confusem.add(score, target)
            confusem1.add(score_1,target)
            confusem2.add(score_2,target)
            confusem3.add(score_3,target)
            # 每迭代一次，训练batch_size张图片  
            _train_loss_1 += now_loss_1
            _train_loss_2 += now_loss_2
            _train_loss_3 += now_loss_3
            if ii%25==0:
                print "------------cm---------------"
                print('cm:%s' % (str(confusem.value)))
                print "------------cm1---------------"
                print('cm1:%s' % (str(confusem1.value)))
                print "------------cm2---------------"
                print('cm2:%s' % (str(confusem2.value)))
                print "------------cm3---------------"
                print('cm3:%s' % (str(confusem3.value)))
            if ii % 120 == 0:  
                # 计算平均train loss  
                train_loss = _train_loss_1 / 120         
                print "step_display : %s train_loss_1 average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss
                _train_loss_1 = 0  
                train_loss = _train_loss_2 / 120         
                print "step_display : %s train_loss_1 average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss
                _train_loss_2 = 0  
                train_loss = _train_loss_3 / 120         
                print "step_display : %s train_loss_1 average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss
                _train_loss_3 = 0  
            if ii%480==479:
                confusem.reset()
                confusem1.reset()
                confusem2.reset()
                confusem3.reset()
   
    del solver  

train_seg()
