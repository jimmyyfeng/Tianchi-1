#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from config import opt
from layers import SingleConv,Deconv


def Deconv(data,cout,kernel_size=2,stride=2,padding=0,is_train=True):
    conv=L.Deconvolution(data,convolution_param=dict(num_output=cout, kernel_size=kernel_size, stride=stride,
        pad=padding, weight_filler=dict(type='xavier')))
    #Deconvolution(data, kernel_size=kernel_size,stride=stride,
    #                        num_output=cout)
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    norm=L.BatchNorm(conv,**kwargs)
    scale=L.Scale(norm,bias_term=True)
    actv=L.ReLU(scale,engine=3)
    return actv


def conv_bn(data,num_output,kernel_size=3,stride=1,padding=1,is_train=True):
    '''
    3D卷积+BatchNorm
    @data：待卷积数据
    @num_output：输出通道
    @kernel_size：卷积核大小
    @stride:步长
    @padding:填充
    Return：Relu激活后的结果
    '''
    conv=L.Convolution(data, kernel_size=kernel_size,stride=stride,pad=padding,
                            num_output=num_output,weight_filler=dict(type='xavier'))
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    conv=L.BatchNorm(conv,**kwargs)
    conv=L.Scale(conv,bias_term=True)
    
    return conv

def SingleConv(data,num_output,kernel_size=3,stride=1,padding=1,is_train=True):
    '''
    3D卷积+BatchNorm+Relu
    @data：待卷积数据
    @num_output：输出通道
    @kernel_size：卷积核大小
    @stride:步长
    @padding:填充
    Return：Relu激活后的结果
    '''
    conv=L.Convolution(data, kernel_size=kernel_size,stride=stride,pad=padding,
                            num_output=num_output,weight_filler=dict(type='xavier'))
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    conv=L.BatchNorm(conv,**kwargs)
    conv=L.Scale(conv,bias_term=True)
    conv=L.ReLU(conv,engine=3)
    return conv

def ResBlock(data,cout,transform = False,z=False):
    # left_=SingleConv(data,cout,kernel_size=3,stride=2,padding=1)
    right_branch1=SingleConv(data,cout,kernel_size=3,stride=1,padding=1)
    right_branch2=SingleConv(data,cout,kernel_size=1,stride=1,padding=0)

    z_kernel_size,z_padding = (1,0) if z else (3,1)
    
    right_branch3 = SingleConv(data,cout,kernel_size=[z_kernel_size,3,3],padding=[z_padding,1,1])
    right_branch3 = SingleConv(right_branch3,cout,kernel_size=3)
    right=[right_branch1,right_branch2,right_branch3]
    right=L.Concat(*right)
    right=conv_bn(right,cout,kernel_size=3,stride=1,padding=1)
    
    if transform:
        data = SingleConv(data,cout)
    return L.ReLU(L.Eltwise(data,right,operation=1,engine=3))

 
def ResDown(data,cout):    
    # left_=SingleConv(data,cout,kernel_size=3,stride=2,padding=1)
    right_branch1=SingleConv(data,cout,kernel_size=3,stride=2,padding=1)
    right_branch2=SingleConv(data,cout,kernel_size=5,stride=2,padding=2)
    
    right=[right_branch1,right_branch2]
    right=L.Concat(*right)
    right=conv_bn(right,cout,kernel_size=3,stride=1,padding=1)

    data = conv_bn(data,cout,kernel_size=3,stride=2,padding=1)
    # data =  L.Pooling(data, kernel_size=3,stride=2,pad=1, pool=P.Pooling.AVE)
    return L.ReLU(L.Eltwise(data,right,operation=1,engine=3))

def ResUp(data,cout):
    right1 = Deconv(data,cout,kernel_size=2,stride=2) 
    right2 = Deconv(data,cout,kernel_size=4,stride=2,padding=1)
    

    right=[right1,right2]
    right=L.Concat(*right)
    right=conv_bn(right,cout,kernel_size=3,stride=1,padding=1)

    data = Deconv(data,cout,kernel_size=2,stride=2)
    return L.ReLU(L.Eltwise(data,right,operation=1,engine=3))

def bn(input,is_train):
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    return L.Scale(L.BatchNorm(input,**kwargs),bias_term=True)

class SegmentationChenYun(object):
    def __init__(self,model_name="seg_chenyun",is_train=True):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.is_train=is_train
        self.model_def="prototxt/chenyun_seg.prototxt"
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'SegDataLayer'

        pydata_params = dict(phase='train', img_root=opt.data_root,
                         batch_size=4,random=True)
        n.data, n.label = L.Python(module='data.SegDataLayer', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

        n.down1 = SingleConv(n.data,64,kernel_size=[3,3,3],stride=[2,2,2])

        n.down2 = ResBlock(n.down1,64)
        n.down3 = ResDown(n.down2,128)
        n.down4 = ResBlock(n.down3,128)
        n.down5 = ResDown(n.down4,256)
        
        n.res = ResBlock(n.down5,256)
        n.res = ResBlock(n.res,256)
        n.res = ResBlock(n.res,256)

        n.up = ResUp(n.res,128)

        cat4 = [n.down4,n.up]
        n.up = L.Concat(*cat4)
        n.up = ResUp(n.up,64)
        
        cat2 = [n.down2,n.up]
        n.up = L.Concat(*cat2)
        n.up = ResUp(n.up,32)

        n.out = L.Convolution(n.up, kernel_size=3,stride=1,pad=1,
                            num_output=1,weight_filler=dict(type='xavier'))
        n.probs=L.Sigmoid(n.out)
        n.probs_=L.Flatten(n.probs)
        n.label_=L.Flatten(n.label)
        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
    
         
  
       
if __name__=="__main__":
    seg=SegmentationChenYun(is_train=True)
    seg.define_model()
        
        
        
        
        
        

        
    
    
    