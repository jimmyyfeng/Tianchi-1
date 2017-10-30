#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from layers import SingleConv,Deconv
class Segmentation(object):
    def __init__(self,model_name="simple_seg",is_train=True):
        self.model_name=model_name
        self.is_train=is_train
        self.model_def='prototxt/simple_seg_train.prototxt'
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'SegDataLayer'

        pydata_params = dict(phase='train', img_root='/home/x/data/datasets/tianchi/',
                         batch_size=4,random=True)
        #data  1,64,64,64
        n.data, n.label = L.Python(module='data.SegDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
        #n.conv1   32,32,32,32
        n.conv1=SingleConv(n.data,32,kernel_size=[3,3,3],stride=[2,2,2],padding=[1,1,1])
        #n.conv2   64,16,16,16
        n.conv2=SingleConv(n.conv1,64,kernel_size=[3,3,3],stride=[2,2,2],padding=[1,1,1])
        #n.conv3  64,8,8,8   
        n.conv3=SingleConv(n.conv2,128,kernel_size=[3,3,3],stride=[2,2,2],padding=[1,1,1])
        #n.conv4 64,4,4,4
        n.conv4=SingleConv(n.conv3,256,kernel_size=[3,3,3],stride=[2,2,2],padding=[1,1,1])
        #n.deconv3 64 8,8,8
        n.deconv3=Deconv(n.conv4,256,128)
        up3=[n.deconv3,n.conv3]
        #n.concat1_3 128,8,8,8
        n.concat1_3=L.Concat(*up3)
        #n.deconv2 64,16,16,16
        n.deconv2=Deconv(n.concat1_3,256,64)
        up2=[n.deconv2,n.conv2]
        #n.concat1_2 128,16,16,16
        n.concat1_2=L.Concat(*up2)
        #n.deconv1 32,32,32,32
        n.deconv1=Deconv(n.concat1_2,128,32)
        up1=[n.deconv1,n.conv1]
        #n.concat1_1 64,32,32,32
        n.concat1_1=L.Concat(*up1)
        #n.concat1_1 32,64,64,64
        n.deconv0=Deconv(n.concat1_1,64,32)
        n.score=L.Convolution(n.deconv0, kernel_size=1,stride=1,pad=0,
                            num_output=1,weight_filler=dict(type='xavier'))
        n.probs=L.Sigmoid(n.score)
        n.probs_=L.Flatten(n.probs)
        n.label_=L.Flatten(n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
        
if __name__=="__main__":
    seg=Segmentation(is_train=True)
    seg.define_model()        
        
        