#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from layers import SingleConv
from config import opt
class MutltiCNN(object):
    def __init__(self,model_name="cls_multi_kernel",is_train=True):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.is_train=is_train
        self.model_def="prototxt/cls_multi_conv_train.prototxt"
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'ClsDataLayer'

        pydata_params = dict(phase='train', data_root=opt.cls_data_root,
                         batch_size=16,ratio=5,augument=True,)
        n.arch1_data,n.arch2_data,n.arch3_data,n.label = L.Python(module='data.ClsDataLayer', layer=pylayer,
            ntop=4, param_str=str(pydata_params))
        n.arch1_conv1=SingleConv(n.arch1_data,64,kernel_size=[3,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch1_conv2=SingleConv(n.arch1_conv1,64,kernel_size=2,stride=2,padding=0)
        n.arch1_conv3=SingleConv(n.arch1_conv2,64,kernel_size=1,stride=1,padding=0)
        n.arch1_conv4=SingleConv(n.arch1_conv3,64,kernel_size=[2,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch1_conv5=SingleConv(n.arch1_conv4,64,kernel_size=[1,4,4],stride=[1,1,1],padding=[0,0,0])
        n.arch1_flat=L.Flatten(n.arch1_conv5)
        n.arch1_fc1=L.InnerProduct(n.arch1_flat, num_output=150,weight_filler=dict(type='xavier'))
        n.fc1_act=L.ReLU(n.arch1_fc1,engine=3)
        n.arch1=L.InnerProduct(n.fc1_act, num_output=2,weight_filler=dict(type='xavier'))
        n.arch1_loss=L.SoftmaxWithLoss(n.arch1, n.label)
        
        n.arch2_conv1=SingleConv(n.arch2_data,64,kernel_size=[3,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch2_conv2=SingleConv(n.arch2_conv1,64,kernel_size=2,stride=2,padding=0)
        n.arch2_conv3=SingleConv(n.arch2_conv2,64,kernel_size=[1,2,2],stride=[1,1,1],padding=[0,0,0])
        n.arch2_conv4=SingleConv(n.arch2_conv3,64,kernel_size=[3,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch2_conv5=SingleConv(n.arch2_conv4,64,kernel_size=[2,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch2_flat=L.Flatten(n.arch2_conv5)
        n.arch2_fc1=L.InnerProduct(n.arch2_flat, num_output=250,weight_filler=dict(type='xavier'))
        n.fc2_act=L.ReLU(n.arch2_fc1,engine=3)
        n.arch2=L.InnerProduct(n.fc2_act, num_output=2,weight_filler=dict(type='xavier'))
        n.arch2_loss=L.SoftmaxWithLoss(n.arch2, n.label)
        
        n.arch3_conv1=SingleConv(n.arch3_data,64,kernel_size=[3,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch3_conv2=SingleConv(n.arch3_conv1,64,kernel_size=2,stride=2,padding=0)
        n.arch3_conv3=SingleConv(n.arch3_conv2,64,kernel_size=[2,2,2],stride=[1,1,1],padding=[0,0,0])
        n.arch3_conv4=SingleConv(n.arch3_conv3,64,kernel_size=[3,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch3_conv5=SingleConv(n.arch3_conv4,64,kernel_size=[3,5,5],stride=[1,1,1],padding=[0,0,0])
        n.arch3_flat=L.Flatten(n.arch3_conv5)
        n.arch3_fc1=L.InnerProduct(n.arch3_flat, num_output=250,weight_filler=dict(type='xavier'))
        n.fc3_act=L.ReLU(n.arch3_fc1,engine=3)
        n.arch3=L.InnerProduct(n.fc3_act, num_output=2,weight_filler=dict(type='xavier'))
        n.arch3_loss=L.SoftmaxWithLoss(n.arch3, n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
if __name__=="__main__":
    seg=MutltiCNN(is_train=True)
    seg.define_model()
        
        
        
        
        
        
        
        
        