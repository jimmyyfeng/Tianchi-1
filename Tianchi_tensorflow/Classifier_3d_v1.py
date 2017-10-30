import tensorflow as tf
from network import Network
training=True
class Classifier(Network):
    def setup(self):
        batch_size=self.batch_size
        keep_prob=self.keep_prob
        (self.feed('data')#(1,48,48,48)
            .conv3d(32,3,1, name='conv1')#(32,48,48,48)
         
            .spatial_red_block(name="srb1",dimensions=3)#(64,24,24,24)
            .res_conc_block(32,name="rcb1",dimensions=3)#(64,24,24,24)
            .spatial_red_block(name="srb2",dimensions=3)#(128,12,12,12)
            .res_conc_block(64,name="rcb2",dimensions=3)#(128,12,12,12)
            .spatial_red_block(name="srb3",dimensions=3)#(256,6,6,6)
            .res_conc_block(128,name="rcb3",dimensions=3)#(256,6,6,6)
            .spatial_red_block(name="srb4",dimensions=3)#(512,3,3,3)
            .res_conc_block(256,name="rcb4",dimensions=3)#(512,3,3,3)
         
            .feat_red(name="fr1",dimensions=3)#(256,3,3,3)
            .res_conc_block(128,name="rcb5",dimensions=3)#(128,3,3,3)
            .feat_red(name="fr2",dimensions=3)#(64,3,3,3)
            .conv3d(2,3,1,name="conv7",padding='VALID',relu=False,norm=False)
            .reshape(2,name="logits")
            .softmax(name='result'))