#encoding:utf-8
from __future__ import print_function
from .module import Module,Flat
import torch as t
import time
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from .layers import Isomorphism_incept,Inception_v2,BasicConv,SingleConv,Isomorphism_incept_1
class Classifier(Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.model_name="iso_classifier"
        self.conv0=BasicConv(1,16)#(40，40，40)
        self.conv1=Isomorphism_incept(16,32) #(40，40，40)
        self.downsample1=nn.MaxPool3d(2)#(20,20,20)
        self.conv2=Isomorphism_incept(32,64)#(20,20,20)
        self.downsample2=nn.MaxPool3d(2)#(10,10,10)
        self.conv3=Isomorphism_incept(64,32) #(10，10，10)
        self.downsample3=nn.MaxPool3d(2)#(5,5,5)
        self.downsample4= nn.Sequential(nn.Conv3d(32,16,(3,3,3)),
                                        nn.BatchNorm3d(16),
                                        nn.ReLU(True),
                                        Flat())#3,3,3
        self.out=nn.Sequential(
            nn.Linear(3*3*3*16,150),
            nn.ReLU(),
            nn.Linear(150,2))
    def forward(self,x):
        conv0=self.conv0(x)
        down1=self.downsample1(self.conv1(conv0))#20
        down2=self.downsample2(self.conv2(down1))#10
        down3=self.downsample3(self.conv3(down2))#5
        down4=self.downsample4(down3)#3
        out=self.out(down4)
        return out
class Classifier_1(Module):
    def __init__(self):
        super(Classifier_1,self).__init__()
        self.model_name="1_iso_classifier"
        self.conv0=BasicConv(1,16)#(40，40，40)
        self.conv1=Isomorphism_incept_1(16,36) #(40，40，40)
        self.downsample1=nn.MaxPool3d(2)#(20,20,20)
        self.conv2=Isomorphism_incept_1(36,72)#(20,20,20)
        self.downsample2=nn.MaxPool3d(2)#(10,10,10)
        self.conv3=Isomorphism_incept_1(72,36) #(10，10，10)
        self.downsample3=nn.MaxPool3d(2)#(5,5,5)
        self.downsample4= nn.Sequential(nn.Conv3d(36,16,(3,3,3)),
                                        nn.BatchNorm3d(16),
                                        nn.ReLU(True),
                                        Flat())#3,3,3
        self.out=nn.Sequential(
            nn.Linear(3*3*3*16,150),
            nn.ReLU(),
            nn.Linear(150,2))
    def forward(self,x):
        conv0=self.conv0(x)
        down1=self.downsample1(self.conv1(conv0))#20
        down2=self.downsample2(self.conv2(down1))#10
        down3=self.downsample3(self.conv3(down2))#5
        down4=self.downsample4(down3)#3
        out=self.out(down4)
        return out