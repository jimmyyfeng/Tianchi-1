import tensorflow as tf
from network import Network
stride=1
training=True
class Unet(Network):
    def setup(self):
        batch_size=self.batch_size
        keep_prob=self.keep_prob
        print batch_size
        print keep_prob
        (self.feed('data')#512*512
             .conv( 32, 3, stride, name='conv1_1')
             .dropout(keep_prob,name="drop1")
             .conv(32, 3, stride, name='conv1_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')#256*256
             .batch_normalization(name="norm1",is_training=training)
         
             .conv(64, 3, stride, name='conv2_1')
             .dropout(keep_prob,name="drop2")
             .conv(64, 3, stride, name='conv2_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')#128*128
             .batch_normalization(name="norm2",is_training=training)
         
             .conv(128, 3, stride, name='conv3_1')
             .dropout(keep_prob,name="drop3")
             .conv( 128, 3, stride, name='conv3_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')#64*64
             .batch_normalization(name="norm3",is_training=training)
         
             .conv(256, 3, stride, name='conv4_1')
             .dropout(keep_prob,name="drop3")
             .conv(256, 3, stride, name='conv4_2')
             # .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')#32,32
             .batch_normalization(name="norm4",is_training=training)
         
             # .conv( 512, 3, 1, name='conv5_1')
             # .conv(512, 3, 1, name='conv5_2')
             .upsample2d([batch_size,128,128,256], 3, 2, name='upsample_1')) #128,128
        (self.feed('conv3_2','upsample_1')
             .merge(axis=3,name='merged_1')
             .conv( 128, 3, stride, name='conv6_1')
             .dropout(keep_prob,name="drop6")
             .conv(128, 3, stride, name='conv6_2')
             .batch_normalization(name="norm6",is_training=training)
             .upsample2d([batch_size,256,256,128], 3, 2, name='upsample_2')) #256,256  
        (self.feed('conv2_2','upsample_2')
             .merge(axis=3,name='merged_2')
             .conv(64, 3, stride, name='conv7_1')
             .dropout(keep_prob,name="drop7")
             .conv(64, 3, stride, name='conv7_2')
             .batch_normalization(name="norm7",is_training=training)
             .upsample2d([batch_size,512,512,64], 3, 2, name='upsample_3')) #256,256
        (self.feed('conv1_2','upsample_3')
             .merge(axis=3,name='merged_3')
             .conv(32, 3, stride, name='conv8_1')
             .dropout(keep_prob,name="drop8")
             .conv(32, 3, stride, name='conv8_2')
             .batch_normalization(name="norm8",is_training=training)
             # .upsample2d( 32, 3, 2, name='upsample_4') #512,512
             .conv(1, 1, stride, name='conv10_1',relu=False)
             .sigmoid(name='result'))
        # (self.feed('conv1_2','upsample_4')
        #      .merge(axis=3,name='merged_3')
        #      .conv(32, 3, 1, name='conv9_1')
        #      .conv(32, 3, 1, name='conv9_2')
        #      .conv(1, 3, 1, name='conv10_1',relu=False)
        #      .sigmoid(name='result'))
         
         
 

