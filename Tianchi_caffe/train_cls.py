#encoding:utf-8
import caffe
import os
import numpy as np
from config import opt
import time
import logging
import psutil
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='myapp.log', #修改log地址为跳板机上的文件
                filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


cls_model_path=None

# 数据读取4核 dataloader.num_workers=4，下标：64-67
# 运行程序是60核，下标：4-63
# PAI平台占用 下标：0-3
os.environ["OMP_NUM_THREADS"]='20'
os.environ["KMP_AFFINITY"]="explicit,1,0,granularity=fine,proclist=[4-23]"

def train_cls():
    solver = caffe.SGDSolver('solver/cls_solver.prototxt')
    if cls_model_path is not None:
        print "loading pre-model......................"
        solver.net.copy_from(cls_model_path)

    # 等价于solver文件中的max_iter，即最大解算次数  
    niter = 1000000  
    # 每隔100次收集一次数据  
    display= 500  

    # 每次测试进行100次解算，10000/100  
    #test_iter = 100  
    # 每500次训练进行一次测试（100次解算），60000/64  
    #test_interval =938  

    #初始化 
    train_loss = np.zeros(niter / display)
    #test_loss = zeros(ceil(niter * 1.0 / test_interval))  
    #test_acc = zeros(ceil(niter * 1.0 / test_interval))  

    # iteration 0，不计入  
    solver.step(1)  

    # 辅助变量  
    _train_loss = 0;# _test_loss = 0; _accuracy = 0  
    # 进行解算  
    for it in range(niter):  
        # 进行一次解算  
        solver.step(1)  
        # 每迭代一次，训练batch_size张图片  
        _train_loss += solver.net.blobs['loss'].data
          
        if os.path.exists("/tmp/debug"):
            import ipdb; ipdb.set_trace()
        if it % display == 0:  
            # 计算平均train loss  
            train_loss[it // display] = _train_loss / display
            print psutil.cpu_times(), psutil.virtual_memory()           
            print "%s train_cls_loss average:"%(time.strftime('%m-%d %H:%M:%S')) , train_loss[it // display]
            _train_loss = 0  
    np.save("train_cls_loss.npy",train_loss)  
        #if it % test_interval == 0:  
        #    for test_it in range(test_iter):  
        #        # 进行一次测试  
        #        solver.test_nets[0].forward()  
        #        # 计算test loss  
        #        _test_loss += solver.test_nets[0].blobs['SoftmaxWithLoss1'].data  
        #        # 计算test accuracy  
        #        _accuracy += solver.test_nets[0].blobs['Accuracy1'].data  
        #    # 计算平均test loss  
        #    test_loss[it / test_interval] = _test_loss / test_iter  
        #    # 计算平均test accuracy  
        #    test_acc[it / test_interval] = _accuracy / test_iter  
        #    _test_loss = 0  
        #    _accuracy = 0  
    del solver  

train_cls()
