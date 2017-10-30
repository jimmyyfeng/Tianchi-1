import tensorflow as tf
from network import Network
training=True
class Unet3D(Network):
    def setup(self):
        batch_size=self.batch_size
        keep_prob=self.keep_prob
        print batch_size
        print keep_prob
        (self.feed('data')#(64,64,64)
            .conv3d(32,3,1, name='conv1')
            .conv3d(32,3,1, name='conv1_1')
            .Inception_v1(32,name="downsample1",dimensions=3)#(32,32,32)
           
         
            .conv3d(64,3,1, name='conv2')#(32,32,32)
            .conv3d(64,3,1, name='conv2_1')#(32,32,32)
            .Inception_v1(64,name="downsample2",dimensions=3)#(16,16,16)
            
        
         
            .conv3d(128,3,1, name='conv3')#(16,16,16)
            .conv3d(128,3,1, name='conv3_1')
            .Inception_v1(128,name="downsample3",dimensions=3)#(8,8,8)
                     
            .conv3d(256,3,1, name='conv4_1')#(8,8,8)
            .conv3d(256,3,1, name='conv4_2')#(8,8,8)
         
           
         
            .upsample3d_v2(256,2,2,name="upsample1"))#(16,16,16)
        (self.feed("upsample1","conv3_1")
            .merge(axis=4,name='merged_1')
            .conv3d(128,3,1, name='conv5_1')
            .conv3d(128,3,1, name='conv5_2')
            .upsample3d_v2(128,2,2,name="upsample2"))#(32,32,32)
        (self.feed("upsample2","conv2_1")
            .merge(axis=4,name='merged_1')
            .conv3d(64,3,1, name='conv6_1')
            .conv3d(64,3,1, name='conv6_2')
            .upsample3d_v2(64,2,2,name="upsample3"))#(64,64,64)
        (self.feed("upsample3","conv1_1")
            .merge(axis=4,name='merged_1')
            .conv3d(32,3,1, name='conv7_1')
            .conv3d(32,3,1, name='conv7_2')
            .conv3d(1,1,1,name="conv_8",relu=False,norm=False)
            .sigmoid(name='result'))
         
          
        
         
         