import tensorflow as tf
from network import Network
training=True
class Classifier(Network):
    def setup(self):
        batch_size=self.batch_size
        keep_prob=self.keep_prob
        (self.feed('data')#(48,48,48)
            .conv3d(16,3,1, name='conv1')
            .ResLy_2(32,1,name="res1",dimensions=3)
            .Inception_v1(32,name="downsample1",dimensions=3)#(24,24,24)
           
         
            .conv3d(32,3,1, name='conv2')#(24,24,24)
            .ResLy_2(64,1,name="res2",dimensions=3)
            .Inception_v1(64,name="downsample2",dimensions=3)#(12,12,12)
            
        
         
            .conv3d(64,3,1, name='conv3')#(12,12,12)
            .ResLy_2(128,1,name="res3",dimensions=3)
            .Inception_v1(128,name="downsample3",dimensions=3)#(6,6,6)
         
            .conv3d(128,3,1, name='conv4')#(6
            .ResLy_2(256,1,name="res4",dimensions=3)
            .Inception_v1(256,name="downsample4",dimensions=3)#3
         
            .conv3d(256,3,1, name='conv5')#(3
            .ResLy_2(512,1,name="res5",dimensions=3)         
            .conv3d(2,3,1,name="conv7",padding='VALID')
            .reshape(2,name="logits")
            .softmax(name='result'))