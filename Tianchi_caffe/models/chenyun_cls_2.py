#encoding:utf-8
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from config import opt



def conv_bn(data,num_output,kernel_size=3,stride=1,padding=1,is_train=True):
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
    z_kernel_size,z_padding = (1,0) if z else (3,1)
    right = SingleConv(data,cout,kernel_size=[z_kernel_size,3,3],padding=[z_padding,1,1])
    right=SingleConv(right,cout,kernel_size=1,stride=1,padding=0)
    if transform:
        data = SingleConv(data,cout)
    return L.ReLU(L.Eltwise(data,right,operation=1,engine=3))

# def downres(data,cout):
#     left = SingleConv(data,3,2,1)
#     right = 

def bn(input,is_train):
    if is_train:
        kwargs={'engine':3}
    else:
        kwargs={'engine':3,'use_global_stats':True}
    return L.Scale(L.BatchNorm(input,**kwargs),bias_term=True)

class MutltiCNN(object):
    def __init__(self,model_name="chenyun_cls_2",is_train=True):
        '''
        @model_def:模型定义.prototext文件路径
        @model_weight:已保存模型参数.caffemodel文件路径
        @model_name:模型名
        '''
        self.model_name=model_name
        self.is_train=is_train
        self.model_def="prototxt/chenyun_cls_2.prototxt"
    def define_model(self):
        n = caffe.NetSpec()
        pylayer = 'ClsDataLayer'

        pydata_params = dict(phase='train', data_root=opt.cls_data_root,
                         batch_size=16,ratio=5,augument=True)
        n.data,n.pre_prob,_,n.label = L.Python(module='data.ClsDataLayer', layer=pylayer,
            ntop=4, param_str=str(pydata_params))

        n.pre_conv=SingleConv(n.data,64,kernel_size=[3,5,5],stride=[1,1,1],padding=0)
        n.pre_conv=SingleConv(n.pre_conv,128,kernel_size=[3,3,3],stride=[2,2,2],padding=0)

        n.res2=ResBlock(n.pre_conv,128,z=True)
        n.res2=ResBlock(n.res2,128)
        
        n.conv = SingleConv(n.res2,128,kernel_size=[2,3,3],stride=[1,1,1],padding=[0,0,0])
        n.conv = SingleConv(n.conv,128,kernel_size=[3,5,5],stride=[1,1,1],padding=0)
        n.pool = L.Pooling(n.conv, kernel_size=[6,3,3],stride=1,pad=0, pool=P.Pooling.MAX)

        # n.features = L.Flatten(n.conv)
        # n.hidden = L.InnerProduct(n.features,num_output=256,weight_filler=dict(type='xavier'))
        n.cls = L.InnerProduct(n.pool,num_output=2,weight_filler=dict(type='xavier'))
        n.prob = L.Softmax(n.cls)
        n.bprob = L.Eltwise(n.prob,n.pre_prob,operation=1,engine=3,coeff=0.5)
        # n.bprob = L.Power(n.bprob,scale=0.5)
        n.loss= L.MultinomialLogisticLoss(n.bprob, n.label)

        with open(self.model_def, 'w') as f:
            f.write(str(n.to_proto()))
if __name__=="__main__":
    seg=MutltiCNN(is_train=True)
    seg.define_model()
        
        
        
        
        
        
        
        
        