import numpy as np
import os
import tensorflow as tf
is_binarised=False
def load(saver,sess,checkpoint_dir,model_dir):
        print(" [*] Reading checkpoints...")

        # model_dir = "unet3d"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False 
def save(saver,sess, checkpoint_dir, steps,model_dir,train_tag=''):
        model_name = model_dir + train_tag + ".model-epoch"
        # model_dir = "classifier"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=steps) 


def pixelwise_cross_entropy(logit, label):
    # N,H,W,num_class = label.get_shape().as_list()
    # assert(num_class==1)
    flat_logit = tf.reshape(logit, [-1])
    flat_label = tf.reshape(label, [-1])
    loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logit, labels=flat_label))
    return loss

def extraLoss1(probs,label):
    return tf.reduce_mean(tf.multiply(1-probs,label))
def extraLoss(probs,label):
    return tf.reduce_mean(tf.multiply(1-label,probs))


def mse(prob,label):
    return tf.reduce_mean(tf.multiply(prob-label,prob-label))

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
def dice_coef_loss(prob, label):
    N,D,H, W, num_class = label.get_shape().as_list()
    assert(num_class==1)

    flat_label = tf.reshape(label, [-1,D*H* W*num_class])
    
    if is_binarised: # theshold to 0 and 1
        prob = tf.cast(tf.round(prob), tf.float32)
    flat_prob  = tf.reshape(prob,  [-1,D*H* W*num_class])
    intersection=tf.reduce_mean(2*tf.multiply(flat_prob,flat_label))+1
    union=tf.reduce_mean(tf.add(flat_prob,flat_label))+1
    loss=1-tf.div(intersection,union)
    # intersection = tf.reduce_sum(tf.multiply(flat_prob,flat_label), axis=1,keep_dims=True)
    # label2 = tf.reduce_sum(tf.multiply(flat_label,flat_label), axis=1,keep_dims=True)
    # prob2  = tf.reduce_sum(tf.multiply(flat_prob,flat_prob) , axis=1,keep_dims=True)
    # union  = tf.reduce_sum(flat_label)+tf.reduce_sum(flat_prob)
    # loss   = 1-tf.reduce_mean( tf.div(1+2*intersection, union+1))
    return loss

