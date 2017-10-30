#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from config import opt
from layers import SingleConv,Deconv
def ResUpModule(data,cout):
    return Deconv(data,cout)
def ResDownModule(data,cout):
    left_=SingleConv(data,cout,kernel_size=3,stride=2,padding=1)
    right_branch1=SingleConv(data,cout,kernel_size=3,stride=1,padding=1)
    right_branch2=SingleConv(data,cout,kernel_size=1,stride=1,padding=0)
    cat_layer=[right_branch1,right_branch2]
    right_branch=L.Concat(*cat_layer)
    right=SingleConv(right_branch,cout,kernel_size=3,stride=2,padding=1)
    return L.Eltwise(left_,right,operation=1)
class Segmentation(object):
    def __init__(self,model_name="seg_res",is_train=True):
        self.model_name=model_name
        self.is_train=is_train
        self.model_def="prototxt/seg_res_train.prototxt"
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'SegDataLayer'

        pydata_params = dict(phase='train', img_root='/home/x',
                         batch_size=4,random=True)
        n.data, n.label = L.Python(module='data.SegDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
        n.conv0=SingleConv(n.data,32)#32 64 64 64
        n.down1=ResDownModule(n.conv0,64)#64 32 32 32
        n.down2=ResDownModule(n.down1,128)#128 16 16 16
        n.down3=ResDownModule(n.down2,128)#128 8 8 8
        n.down4=ResDownModule(n.down3,128)#128 4 4 4
        
        n.up3=ResUpModule(n.down4,128)#128 8 8 8
        cat3=[n.down3,n.up3]
        n.cat3=L.Concat(*cat3)#256 8 8 8
        
        n.up2=ResUpModule(n.cat3,128)#128 16 16 16
        cat2=[n.down2,n.up2]
        n.cat2=L.Concat(*cat2)#256 16 16 16
        
        n.up1=ResUpModule(n.cat2,128)#128,32,32,32
        cat1=[n.down1,n.up1]
        n.cat1=L.Concat(*cat1)#196,32,32,32
        
        n.up0=ResUpModule(n.cat1,32)#32,64,64,64
        n.score=L.Convolution(n.up0, kernel_size=1,stride=1,pad=0,
                            num_output=1,weight_filler=dict(type='xavier'))
        n.probs=L.Sigmoid(n.score)
        n.probs_=L.Flatten(n.probs)
        n.label_=L.Flatten(n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
if __name__=="__main__":
    seg=Segmentation(is_train=True)
    seg.define_model()

        
        
        
        
    
    
    