#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
#from data import SegDataLayer
from layers import SingleConv,BasicConv,Inception_v1,Inception_v2,Deconv
class Config:
    model_def="./segmentation_train.prototxt"#模型定义.prototext文件路径
    model_weight=None#已保存模型参数.caffemodel文件路径
    model_solver=None#Solver定义文件保存路径
    batch_size=4#训练batch_size大小
    lmdb_path='../mnist/mnist_train_lmdb'#数据路径
opt=Config()
class Segmentation(object):
    def __init__(self,model_name="seg"):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.model_def=opt.model_def
        self.model_weight=opt.model_weight
        self.model_solver=opt.model_solver
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'SegDataLayer'
        pydata_params = dict(phase='train', img_root='/home/x/data/datasets/tianchi/',
                             batch_size=4,random=True)
        n.data, n.label = L.Python(module='data.SegDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
        n.conv1=BasicConv(n.data,32)#(64,64,64)
        n.conv9=L.Convolution(n.conv1, kernel_size=1,stride=1,pad=0,
                            num_output=1,weight_filler=dict(type='xavier'))
        n.loss=L.SoftmaxWithLoss(n.conv9, n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
    
        
                
    def define_solver(self):
        pass
        
if __name__=="__main__":
    seg=Segmentation()
    seg.define_model()
    