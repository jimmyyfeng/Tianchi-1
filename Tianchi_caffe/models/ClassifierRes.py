#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from config import opt
from layers import SingleConv,Deconv
def ResDownModule(data,cout):
    left_=SingleConv(data,cout,kernel_size=3,stride=2,padding=1)
    right_branch1=SingleConv(data,cout,kernel_size=3,stride=1,padding=1)
    right_branch2=SingleConv(data,cout,kernel_size=1,stride=1,padding=0)
    cat_layer=[right_branch1,right_branch2]
    right_branch=L.Concat(*cat_layer)
    right=SingleConv(right_branch,cout,kernel_size=3,stride=2,padding=1)
    return L.Eltwise(left_,right,operation=1)
class Classifier(object):
    def __init__(self,model_name="cls_res",is_train=True):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.is_train=is_train
        self.model_def="prototxt/cls_res_train.prototxt"
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'ClsDataLayer'

        pydata_params = dict(phase='train', data_root=opt.cls_data_root,
                         batch_size=16,ratio=5,augument=True,)
        #n.data  40,40,40
        n.data,n.label = L.Python(module='data.ClsDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
        # 20,20 20
        n.conv1=ResDownModule(n.data,32)
        #10,10,10
        n.conv2=ResDownModule(n.conv1,64)
        #5,5,5
        n.conv3=ResDownModule(n.conv2,128)
        #3,3,3
        n.conv4=SingleConv(n.conv3,128,kernel_size=3,stride=1,padding=0)
        n.flatten=L.Flatten(n.conv4)#128*3*3*3
        n.fc1=L.InnerProduct(n.flatten, num_output=250,weight_filler=dict(type='xavier'))
        n.fc1_act=L.ReLU(n.fc1,engine=3)
        n.score=L.InnerProduct(n.fc1_act, num_output=2,weight_filler=dict(type='xavier'))
        n.loss=L.SoftmaxWithLoss(n.score, n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
if __name__=="__main__":
    seg=Classifier(is_train=True)
    seg.define_model()
        
        
        
        
        
        