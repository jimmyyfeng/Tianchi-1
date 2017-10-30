#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from config import opt
from data import ClsDataLayer
from layers import SingleConv,BasicConv,Isomorphism_incept_1
class Classifier(object):
    def __init__(self,model_name="cls_multi_kernel",is_train=True):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.is_train=is_train
        self.model_def="prototxt/cls_multi_kernel_train.prototxt"
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'ClsDataLayer'

        pydata_params = dict(phase='train', data_root=opt.cls_data_root,
                         batch_size=16,ratio=5,augument=True,)
        n.data, n.label = L.Python(module='data.ClsDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
        n.conv0=BasicConv(n.data,16)#(40，40，40)
        n.conv1=Isomorphism_incept_1(n.conv0,36) #(40，40，40)
        n.downsample1=L.Pooling(n.conv1, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
        n.conv2=Isomorphism_incept_1(n.downsample1,72)#(20,20,20)
        n.downsample2=L.Pooling(n.conv2, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
        n.conv3=Isomorphism_incept_1(n.downsample2,36) #(10，10，10)
        n.downsample3=L.Pooling(n.conv3, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)#(5，5，5)
        n.conv4=L.Convolution(n.downsample3, kernel_size=3,stride=1,pad=1,#(3,3,3)
                            num_output=16,weight_filler=dict(type='xavier'))
        n.flatten=L.Flatten(n.conv4)#(16*3*3*3)
        
        n.fc1= L.InnerProduct(n.flatten, num_output=150,weight_filler=dict(type='xavier'))
        n.fc1_act=L.ReLU(n.fc1,engine=3)
        n.score= L.InnerProduct(n.fc1_act, num_output=2,weight_filler=dict(type='xavier'))
        n.loss = L.SoftmaxWithLoss(n.score, n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
        
if __name__=="__main__":
    seg=Classifier(is_train=True)
    seg.define_model()
