#encoding:utf-8
#import torch as t
#global t
import os
import sys
from config import opt
if opt.online:
    os.environ["PYTHONPATH"]="%s:%s"%("/home/caffe/python","/workspace/pai")
    sys.path.append("/home/caffe/python")
    sys.path.append("/workspace/pai")
#import ipdb;ipdb.set_trace()
import numpy as np
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



seg_model_path=None#opt.model_seg_pre_weight

# 数据读取4核 dataloader.num_workers=4，下标：64-67
# 运行程序是60核，下标：4-63
# PAI平台占用 下标：0-3
if opt.online==False:
    os.environ["OMP_NUM_THREADS"]='55'
    os.environ["KMP_AFFINITY"]="explicit,1,0,granularity=fine,proclist=[4-58]"
else:
    os.environ["OMP_NUM_THREADS"]='60'
    os.environ["KMP_AFFINITY"]="explicit,1,0,granularity=fine,proclist=[4-63]"
def train_seg():
    #subprocess.Popen('top')
    solver = caffe.AdamSolver('solver/seg_solver.prototxt')
    if seg_model_path is not None:
        print "loading pre-model.",seg_model_path
        solver.net.copy_from(seg_model_path)

    # 等价于solver文件中的max_iter，即最大解算次数  
    niter = 10000  
    # 每隔100次收集一次数据  
    display= 50  if opt.online==False else 20
    display_=622
    

    #初始化 
    train_loss = np.zeros(niter / display)
    train_loss_ = np.zeros(niter / display_)
    
    solver.step(1)  

    # 辅助变量  
    _train_loss = 0;# _test_loss = 0; _accuracy = 0  
    _all_loss=0
    # 进行解算  
    #loss_meter = tnt.meter.AverageValueMeter()
    #loss_meter.reset()
    for it in range(niter):  
        #import ipdb;ipdb.set_trace()
        # 进行一次解算  
        solver.step(1) 
        now_loss=solver.net.blobs["loss"].data 
        #loss_meter.add(now_loss[0])
        # 每迭代一次，训练batch_size张图片  
        _train_loss += now_loss
        _all_loss +=now_loss
        #print "setp: ",it,"time: ",time.strftime('%m-%d %H:%M:%S') ,"  train_loss average:",_all_loss/(it%display_+1) 
        if os.path.exists("/tmp/debug"):
            import ipdb; ipdb.set_trace()
        if it % display_ == 0: 
            train_loss_[it // display_] = _all_loss / display_         
            print "step 622: %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss_[it // display_]
            _all_loss=0
            #loss_meter.reset()
        if it % display == 0:  
            # 计算平均train loss  
            train_loss[it // display] = _train_loss / display
            print psutil.cpu_percent(percpu=True), psutil.virtual_memory()           
            print "step 50: %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss[it // display]
            _train_loss = 0  
            #vis_plots = {'loss':loss_meter.value()[0],'it':it}
            #vis.plot_many(vis_plots)

            #iik=np.random.randint(0,4,1)[0]
            #image=t.from_numpy(solver.net.blobs["data"].data[iik] )
            #probs=t.from_numpy(solver.net.blobs["probs"].data[iik])
            #mask=t.from_numpy(solver.net.blobs["label"].data[iik] )
            #print image.size(),probs.size(), mask.size()
            #vis.vis.histogram(
             #      probs.view(-1), win=u'output_hist', opts=dict(title='output_hist'))
                #！TODO: tell 代成 make 1/3 和1 ，而不是1和3
            #vis_imgs = {'input':image,'mask':mask,'output':probs}
            #vis.img_grid_many(vis_imgs)
    np.save("train_loss.npy",train_loss)  
    del solver  

cls_model_path="snashots/cls_multi_kernel_iter_960.caffemodel"    
def train_cls():
    #subprocess.Popen('top')
    solver = caffe.AdamSolver('solver/cls_solver.prototxt')
    if cls_model_path is not None:
        print "loading pre-model.",cls_model_path
        solver.net.copy_from(cls_model_path)
  
    niter = 100000  
    # 每隔100次收集一次数据  
    display= 480  if opt.online==False else 20
    # 每次测试进行100次解算，10000/100  
    #test_iter = 60 
    # 每500次训练进行一次测试（100次解算），60000/64  
    #test_interval =240  

    #初始化 
    train_loss = np.zeros(niter / display)
    train_acc=np.zeros(niter / display)
    #test_loss = np.zeros(niter  / test_interval)  
    #test_acc = np.zeros(niter  / test_interval)

    # iteration 0，不计入  
    solver.step(1)  

    # 辅助变量  
    _train_loss = 0;_test_loss = 0; _accuracy = 0  
    # 进行解算  
    #loss_meter = tnt.meter.AverageValueMeter()
    #loss_meter.reset()
    confusem = ConfusionMeter(2)
    confusem.reset()
    for it in range(niter):  
        #import ipdb;ipdb.set_trace()
        # 进行一次解算  
        solver.step(1) 
        now_loss=solver.net.blobs["loss"].data 
        score=solver.net.blobs["score"].data 
        target=solver.net.blobs["label"].data 
        confusem.add(score, target)
        # 每迭代一次，训练batch_size张图片  
        _train_loss += now_loss
        _accuracy += solver.net.blobs['Accuracy1'].data
        if os.path.exists("/tmp/debug"):
            import ipdb; ipdb.set_trace()
        if it%5==0:
            print('cm:%s' % (str(confusem.value)))
        if it % display == 0:  
            # 计算平均train loss  
            train_loss[it // display] = _train_loss / display         
            print "step_display : %s train_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss[it // display]
            _train_loss = 0  
            train_acc[it // display] = _accuracy / display  
            print "step_acc : %s train_acc average:"%(time.strftime('%m-%d %H:%M:%S')) , train_acc[it // display]
            _accuracy=0
            confusem.reset()
            
    del solver  

train_seg()
