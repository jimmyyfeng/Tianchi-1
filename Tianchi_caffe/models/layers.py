#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
import math
def reset_parameters(kernel_size,in_channel):
        n = in_channel
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        return -1*stdv,stdv
def Deconv(data,cout,kernel_size=2,stride=2,is_train=True):
    conv=L.Deconvolution(data,convolution_param=dict(num_output=cout, kernel_size=kernel_size, stride=stride,
        weight_filler=dict(type='xavier')))
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
    norm=L.BatchNorm(conv,**kwargs)
    scale=L.Scale(norm,bias_term=True)
    actv=L.ReLU(scale,engine=3)
    return actv
def BasicConv(data,num_output,kernel_size=3,stride=1,padding=1,is_train=True):
    '''
    两次3D卷积+BatchNorm+Relu
    @data：待卷积数据
    @num_output：输出通道
    @kernel_size：卷积核大小
    @stride:步长
    Return：Relu激活后的结果
    '''
    conv=L.Convolution(data, kernel_size=kernel_size,stride=stride,pad=padding,
                            num_output=num_output,weight_filler=dict(type='xavier'))
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    norm=L.BatchNorm(conv,**kwargs)
    scale=L.Scale(norm,bias_term=True)
    actv=L.ReLU(scale,engine=3)
    conv2=L.Convolution(actv, kernel_size=kernel_size,stride=stride,pad=padding,
                            num_output=num_output,weight_filler=dict(type='xavier'))
    norm2=L.BatchNorm(conv2,**kwargs)
    scale2=L.Scale(norm2,bias_term=True)
    actv2=L.ReLU(scale2,engine=3)
    return actv2

def Inception_v2(data,cin,co,relu=True,norm=True,is_train=True):
    '''
    一种Inception结构，使输入与输出大小保持不变。
    @data:待卷积数据
    @cin：输入通道数
    @cout：输出通道数
    @relu：输出是否进行Relu激活
    @norm：输出是否BatchNormalization
    '''
    assert(co%4==0)
    cos=[co/4]*4
    #分支1:1*1卷积，步长为1
    if is_train:
        kwargs=kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    branch1=L.Convolution(data, kernel_size=1,stride=1,pad=0,
                            num_output=cos[0],weight_filler=dict(type='xavier'))
    #分支2：Conv+BN+RELU+Conv
    branch2_conv1=L.Convolution(data, kernel_size=1,stride=1,pad=0,
                            num_output=2*cos[1],weight_filler=dict(type='xavier'))
    branch2_norm1=L.BatchNorm(branch2_conv1,**kwargs)
    branch2_scale1=L.Scale(branch2_norm1,bias_term=True)
    branch2_relu1=L.ReLU(branch2_scale1,engine=3)
    branch2=L.Convolution(branch2_relu1, kernel_size=3,stride=1,pad=1,
                            num_output=cos[1],weight_filler=dict(type='xavier'))
    #分支3：Conv(1,1,0)+BN+RELU+Conv(5,1,2)
    branch3_conv1=L.Convolution(data, kernel_size=1,stride=1,pad=0,
                            num_output=2*cos[2],weight_filler=dict(type='xavier'))
    branch3_norm1=L.BatchNorm(branch3_conv1,**kwargs)
    branch3_scale1=L.Scale(branch3_norm1,bias_term=True)
    branch3_relu1=L.ReLU(branch3_scale1,engine=3)
    branch3=L.Convolution(branch3_relu1, kernel_size=5,stride=1,pad=2,
                            num_output=cos[2],weight_filler=dict(type='xavier'))
    #分支4：MaxPool+Conv
    branch4_pool1=L.Pooling(data, kernel_size=3,
                        stride=1,pad=1, pool=P.Pooling.MAX)
    branch4=L.Convolution(branch4_pool1, kernel_size=1,stride=1,pad=0,
                            num_output=cos[3],weight_filler=dict(type='xavier'))
    
    #concat branch1,branch2,branch3,branch4
    bottom_layers = [branch1,branch2,branch3,branch4]
    result=L.Concat(*bottom_layers)
    if norm:
        result=L.BatchNorm(result,**kwargs)
        result=L.Scale(result,bias_term=True)
    if relu:
        result=L.ReLU(result,engine=3)
    return result

def Inception_v1(data,cin,co,relu=True,norm=True,is_train=True):
    '''
    一种Inception结构，使输入与输出大小减半。
    @data:待卷积数据
    @cin：输入通道数
    @cout：输出通道数
    @relu：输出是否进行Relu激活
    @norm：输出是否BatchNormalization
    '''
    assert(co%4==0)
    cos=[co/4]*4
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    #分支1:1*1卷积，步长为1
    branch1=L.Convolution(data, kernel_size=1,stride=2,pad=0,
                            num_output=cos[0],weight_filler=dict(type='xavier'))
    #分支2：Conv+BN+RELU+Conv
    branch2_conv1=L.Convolution(data, kernel_size=1,stride=1,pad=0,
                            num_output=2*cos[1],weight_filler=dict(type='xavier'))
    branch2_norm1=L.BatchNorm(branch2_conv1,**kwargs)
    branch2_scale1=L.Scale(branch2_norm1,bias_term=True)
    branch2_relu1=L.ReLU(branch2_scale1,engine=3)
    branch2=L.Convolution(branch2_relu1, kernel_size=3,stride=2,pad=1,
                            num_output=cos[1],weight_filler=dict(type='xavier'))
    #分支3：Conv(1,1,0)+BN+RELU+Conv(5,1,2)
    branch3_conv1=L.Convolution(data, kernel_size=1,stride=1,pad=0,
                            num_output=2*cos[2],weight_filler=dict(type='xavier'))
    branch3_norm1=L.BatchNorm(branch3_conv1,**kwargs)
    branch3_scale1=L.Scale(branch3_norm1,bias_term=True)
    branch3_relu1=L.ReLU(branch3_scale1,engine=3)
    branch3=L.Convolution(branch3_relu1, kernel_size=5,stride=2,pad=2,
                            num_output=cos[2],weight_filler=dict(type='xavier'))
    #分支4：MaxPool+Conv
    branch4_pool1=L.Pooling(data, kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    branch4=L.Convolution(branch4_pool1, kernel_size=1,stride=1,pad=0,
                            num_output=cos[3],weight_filler=dict(type='xavier'))
    
    #concat branch1,branch2,branch3,branch4
    bottom_layers = [branch1,branch2,branch3,branch4]
    result=L.Concat(*bottom_layers)
    if norm:
        result=L.BatchNorm(result,**kwargs)
        result=L.Scale(result,bias_term=True)
    if relu:
        result=L.ReLU(result,engine=3)
    return result
    
def Isomorphism_incept_1(data,co,relu=True,norm=True,is_train=True):
    '''
    异构卷积核，根据节点ZYX spacing设计相应尺寸，kernel size，Z/X的比例靠近spacingZ/spacingX
    保持大小不变
    '''
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    assert(co%6==0)
    cos=[co/6]*6

    branch1 =L.Convolution(data, kernel_size=[1,1,1],stride=[1,1,1],pad=[0,0,0],
                            num_output=cos[0],weight_filler=dict(type='xavier'))  
    branch2 =L.Convolution(data, kernel_size=[1,5,5],stride=[1,1,1],pad=[0,2,2],
                            num_output=cos[1],weight_filler=dict(type='xavier')) 
    branch3 =L.Convolution(data, kernel_size=[1,3,3],stride=[1,1,1],pad=[0,1,1],
                            num_output=cos[2],weight_filler=dict(type='xavier')) 
    branch4 =L.Convolution(data, kernel_size=[5,7,7],stride=[1,1,1],pad=[2,3,3],
                            num_output=cos[3],weight_filler=dict(type='xavier'))   
    branch5 =L.Convolution(data, kernel_size=[3,5,5],stride=[1,1,1],pad=[1,2,2],
                            num_output=cos[4],weight_filler=dict(type='xavier'))  
    branch6 =L.Convolution(data, kernel_size=[7,5,5],stride=[1,1,1],pad=[3,2,2],
                            num_output=cos[5],weight_filler=dict(type='xavier'))  
    bottom_layers = [branch1,branch2,branch3,branch4,branch5,branch6]
    result=L.Concat(*bottom_layers)
    if norm:
        result=L.BatchNorm(result,**kwargs)
        result=L.Scale(result,bias_term=True)
    if relu:
        result=L.ReLU(result,engine=3)
   
    return result
    
    
    
    
    
    
    
    
    
