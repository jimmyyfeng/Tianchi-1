#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from layers import SingleConv,BasicConv,Inception_v1,Inception_v2,Deconv
class Config:
    model_def="prototxt/classifier_train.prototxt"#模型定义.prototext文件路径
    model_weight=None#已保存模型参数.caffemodel文件路径
    model_solver=None#Solver定义文件保存路径
    batch_size=4#训练batch_size大小
    lmdb_path='../mnist/mnist_train_lmdb'#数据路径
    nodule_cubic='/mnt/7/train_nodule_cubic/'#从训练样本上切下的结点立方体保存路径
    candidate_cubic='/mnt/7/0705_train_48_64_candidate/'#从训练样本上切下的候选结点立方体保存路径
    ratio=5
        
opt=Config()
kwargs={'engine':1}

class Classifier(object):
    def __init__(self,model_name="classifier"):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.model_def=opt.model_def
        self.model_weight=opt.model_weight
        self.model_solver=opt.model_solver
        self.nodule_cubic=opt.nodule_cubic
        self.candidate_cubic=opt.candidate_cubic
        self.ratio=opt.ratio
        
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'ClsDataLayer'
        pydata_params = dict(phase='train', nodule_cubic=self.nodule_cubic,candidate_cubic=self.candidate_cubic, ratio=self.ratio,
        #pydata_params = dict(phase='train', img_root='/workspace/pai/data/',
                             batch_size=4,crop_size=[40,40,40],random=True)
        n.data, n.label = L.Python(module='data.ClsDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
        n.conv1=BasicConv(n.data,32)#(40,40,40)
        n.downsample1=Inception_v1(n.conv1,32,32)#(20,20,20)
        n.conv2=BasicConv(n.downsample1,64)
        n.downsample2=Inception_v1(n.conv2,64,64)#(10,10,10)
        n.conv3=BasicConv(n.downsample2,64)
        n.downsample3=Inception_v1(n.conv3,64,64)#(5,5,5)
        n.conv4=BasicConv(n.downsample3,64)
        
        n.conv5=L.Convolution(n.conv4, kernel_size=3,stride=1,pad=0,num_output=16,weight_filler=dict(type='xavier'))
        n.bn5=L.BatchNorm(n.conv5,**kwargs)
        n.re5=L.ReLU(n.bn5,**kwargs)
        
        n.fc6=L.InnerProduct(n.re5, num_output=150,weight_filler=dict(type='xavier'))
        n.re6=L.ReLU(n.fc6,**kwargs)
        
        n.fc7=L.InnerProduct(n.re6, num_output=2,weight_filler=dict(type='xavier'))
        
        
        n.loss=L.SoftmaxWithLoss(n.fc7,n.label)
        #n.loss=L.Python(module='DiceLoss', layer="DiceLossLayer",
        #    ntop=1, bottom=[n.probs,n.label])
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
    
        
                
    def define_solver(self):
        pass
        
if __name__=="__main__":
    classifier=Classifier()
    classifier.define_model()
    