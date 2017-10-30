#encoding:utf-8
import numpy as np


class ConfusionMeter(object):
    def __init__(self,class_num=2):
        self.class_num=class_num
        self.value=np.zeros([self.class_num,self.class_num])
    def add(self,predict,target):
        '''
        predict:分类网络的输出，batch*class_num
        target:目标，batch*1
        '''
        assert predict.shape[0]==target.shape[0]
        batch_size=predict.shape[0]
        if len(target.shape)>1:
            target=target.reshape(predict.shape[0])
        #预测的类别
        predict_arg=predict.argmax(axis=1)
        #target每个类的数目
        per_class_target=np.array([ sum(target==r) for r in range(self.class_num)])
        
        for i in range(self.class_num):
            for j in range(self.class_num):
                self.value[i,j] +=sum((target==i)&(predict_arg==j))
    def reset(self):
        self.value=np.zeros([self.class_num,self.class_num])
        

            
            
        