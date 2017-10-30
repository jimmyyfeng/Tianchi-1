#encoding:utf-8
import caffe
import numpy as np


class DiceLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        # loss output is scalar
        top[0].reshape(1)
        self.diff=np.zeros_like(bottom[0].data,dtype=np.float32)
        #分母  与输入大小相同矩阵,逐元素和
        self.down= 0
        self.up=0 #分子，与输入大小相同矩阵,逐元素乘积
    def forward(self, bottom, top):
        self.diff[...]=bottom[1].data
        self.up=np.sum(bottom[0].data*bottom[1].data)
        self.down=np.sum(bottom[0].data+bottom[1].data)
        self.dice=2*self.up/self.down
        loss=1-self.dice
        top[0].data[...]=loss

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("label not diff")
        elif propagate_down[0]:
            bottom[0].diff[...] = -1*(2*self.diff-self.dice)/self.down#(-2.*self.diff+self.dice)/self.sum
        else:
            raise Exception("no diff")
