#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from config import opt
#from data import SegDataLayer
from layers import SingleConv,BasicConv,Inception_v1,Inception_v2,Deconv


class Segmentation(object):
    def __init__(self,model_name="seg",is_train=True):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.is_train=is_train
        self.model_def=opt.model_def_seg_train
        self.model_weight=opt.model_seg_pre_weight
        self.model_solver=opt.model_seg_solver
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'SegDataLayer'

        pydata_params = dict(phase='train', img_root=opt.data_root,
                         batch_size=4,random=True)
        n.data, n.label = L.Python(module='data.SegDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
        n.conv1=BasicConv(n.data,32,is_train=self.is_train)#(64,64,64)
        n.downsample1=Inception_v1(n.conv1,32,32,is_train=self.is_train)#(32,32,32)
        n.conv2=BasicConv(n.downsample1,64,is_train=self.is_train)
        n.downsample2=Inception_v1(n.conv2,64,64,is_train=self.is_train)#(16,16,16)
        n.conv3=BasicConv(n.downsample2,128,is_train=self.is_train)
        n.downsample3=Inception_v1(n.conv3,128,128,is_train=self.is_train)#(8,8,8)
        n.conv4=BasicConv(n.downsample3,256,is_train=self.is_train)
        n.downsample4=Inception_v1(n.conv4,256,256,is_train=self.is_train)#(4,4,4)
        
        n.conv4_=SingleConv(n.downsample4,128,is_train=self.is_train)
        n.incept4=Inception_v2(n.conv4_,128,128,is_train=self.is_train)
        n.deconv4=Deconv(n.incept4,128,128,is_train=self.is_train)#(8,8,8)
        up4=[n.deconv4,n.conv4]
        n.concat1_4=L.Concat(*up4)
        
        n.conv5=SingleConv(n.concat1_4,128,is_train=self.is_train)
        n.incept5=Inception_v2(n.conv5,128,128,is_train=self.is_train)
        n.deconv5=Deconv(n.incept5,128,128,is_train=self.is_train)#(16,16,16)
        up5=[n.deconv5,n.conv3]
        n.concat1_5=L.Concat(*up5)
        
        n.conv6=SingleConv(n.concat1_5,64,is_train=self.is_train)
        n.incept6=Inception_v2(n.conv6,64,64,is_train=self.is_train)
        n.deconv6=Deconv(n.incept6,64,64,is_train=self.is_train)#(32,32,32)
        up6=[n.deconv6,n.conv2]
        n.concat1_6=L.Concat(*up6)
        
        n.conv7=SingleConv(n.concat1_6,32,is_train=self.is_train)
        n.incept7=Inception_v2(n.conv7,32,32,is_train=self.is_train)
        n.deconv7=Deconv(n.incept7,32,32,is_train=self.is_train)#(64,64,64)
        up7=[n.deconv7,n.conv1]
        n.concat1_7=L.Concat(*up7)
        
        n.conv8=SingleConv(n.concat1_7,32,is_train=self.is_train)
        n.incept8=Inception_v2(n.conv8,32,32,is_train=self.is_train)
        n.conv9=L.Convolution(n.incept8, kernel_size=1,stride=1,pad=0,
                            num_output=1,weight_filler=dict(type='xavier'))
        n.probs=L.Sigmoid(n.conv9)
        n.probs_=L.Flatten(n.probs)
        n.label_=L.Flatten(n.label)
        #n.loss=L.SoftmaxWithLoss(n.conv9,n.label)
        #n.loss=L.Python(module='DiceLoss', layer="DiceLossLayer",
        #    ntop=1, bottom=[n.probs,n.label])
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
    
        
                
    def define_solver(self):
        pass
        
if __name__=="__main__":
    seg=Segmentation(is_train=True)
    seg.define_model()
    