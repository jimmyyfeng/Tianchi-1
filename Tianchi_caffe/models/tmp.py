#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
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
        n.data, n.label = L.Data(batch_size=opt.batch_size,
                             backend=P.Data.LMDB, source=opt.lmdb_path,
                             transform_param=dict(scale=1. / 255), ntop=2)
        n.conv1=BasicConv(n.data,32)#(64,64,64)
        n.downsample1=Inception_v1(n.conv1,32,32)#(32,32,32,32)
        n.conv2=BasicConv(n.downsample1,64)
        n.downsample2=Inception_v1(n.conv2,64,64)#(16,16,16,16)
        n.conv3=BasicConv(n.downsample2,128)
        n.downsample3=Inception_v1(n.conv3,128,128)#(8,8,8)
        n.conv4=BasicConv(n.data,256)
        n.downsample4=Inception_v1(n.conv4,256,256)#(4,4,4)
        
        n.conv4_=SingleConv(n.downsample4,128)
        n.incept4=Inception_v2(n.conv4_,128,128)
        n.deconv4=Deconv(n.incept4,128)#(8,8,8)
        up4=[n.deconv4,n.conv4]
        n.concat4=L.Concat(*up4,in_place=True)
        
        n.conv5=SingleConv(n.concat4,128)
        n.incept5=Inception_v2(n.conv5,128,128)
        n.deconv5=Deconv(n.incept5,128,128)#(16,16,16)
        up5=[n.deconv5,n.conv3]
        n.concat5=L.Concat(*up5, in_place=True)
        
        n.conv6=SingleConv(n.concat5,64)
        n.incept6=Inception_v2(n.conv6,64,64)
        n.deconv6=Deconv(n.incept6,64,64)#(32,32,32)
        up6=[n.deconv6,n.conv2]
        n.concat6=L.Concat(*up6,in_place=True)
        
        n.conv7=SingleConv(n.concat6,32)
        n.incept7=Inception_v2(n.conv7,32,32)
        n.deconv7=Deconv(n.incept7,32,32)#(64,64,64)
        up7=[n.deconv7,n.conv1]
        n.concat7=L.Concat(*up7, in_place=True)
        
        n.conv8=SingleConv(n.concat7,32)
        n.incept8=Inception_v2(n.conv8,32,32)
        n.conv9=L.Convolution(n.incept8, kernel_size=1,stride=1,pad=0,
                            num_output=1,weight_filler=dict(type='xavier'))
        n.loss=L.SoftmaxWithLoss(n.conv9, n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
    
        
                
    def define_solver(self):
        pass
        
if __name__=="__main__":
    seg=Segmentation()
    seg.define_model()
    