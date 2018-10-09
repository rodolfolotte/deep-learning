from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf
import numpy as np
import encoder
import decoder 

def buildEncoder(hypes, images, train):        
    vgg_fcn = encoder.VGG16FCN(vgg16_npy_path=hypes['data']['weights_path'])
    vgg_fcn.wd = float(hypes['arch']['wd'])

    logging.info(".....Creating encoder (VGG16 model)")              
    vgg_fcn.build(images, train=train, num_classes=hypes['arch']['num_classes'], random_init_fc8=True)

    logits = {}
    logits['images'] = images
    logits['fcn_in'] = vgg_fcn.pool5    
    logits['feed2'] = vgg_fcn.pool4
    logits['feed4'] = vgg_fcn.pool3    
    logits['fcn_logits'] = vgg_fcn.upscore32
    logits['deep_feat'] = vgg_fcn.pool5
    logits['early_feat'] = vgg_fcn.conv4_3

    return logits


def buildDecoder(hypes, logits, train=True, skip=True):
    fcn_in = logits['fcn_in']
    num_classes = hypes['arch']['num_classes']
    sd = 1
    logging.info(".....Creating decoder")
    
    he_init = tf.contrib.layers.variance_scaling_initializer()
    l2_regularizer = tf.contrib.layers.l2_regularizer(hypes['arch']['wd'])

    score_fr = tf.layers.conv2d(
        fcn_in, kernel_size=[1, 1], filters=num_classes, padding='SAME',
        name='score_fr', kernel_initializer=he_init,
        kernel_regularizer=l2_regularizer)

    # addToSummary(score_fr)
    
    upscore2 = decoder.upscoreLayer(
        score_fr, upshape=tf.shape(logits['feed2']),
        num_classes=num_classes, name='upscore2', ksize=4, stride=2)

    he_init2 = tf.contrib.layers.variance_scaling_initializer(factor=2.0 * sd)
    
    score_feed2 = tf.layers.conv2d(
        logits['feed2'], kernel_size=[1, 1], filters=num_classes,
        padding='SAME', name='score_feed2', kernel_initializer=he_init2,
        kernel_regularizer=l2_regularizer)

    # addToSummary(score_feed2)

    if skip:        
        fuse_feed2 = tf.add(upscore2, score_feed2)
    else:
        fuse_feed2 = upscore2
        fuse_feed2.set_shape(score_feed2.shape)
    
    upscore4 = decoder.upscoreLayer(
        fuse_feed2, upshape=tf.shape(logits['feed4']),
        num_classes=num_classes, name='upscore4', ksize=4, stride=2)

    he_init4 = tf.contrib.layers.variance_scaling_initializer(factor=2.0 * sd * sd)
    
    score_feed4 = tf.layers.conv2d(
        logits['feed4'], kernel_size=[1, 1], filters=num_classes,
        padding='SAME', name='score_feed4', kernel_initializer=he_init4,
        kernel_regularizer=l2_regularizer)

    # addToSummary(score_feed4)

    if skip:        
        fuse_pool3 = tf.add(upscore4, score_feed4)
    else:
        fuse_pool3 = upscore4
        fuse_pool3.set_shape(score_feed4.shape)

    upscore32 = decoder.upscoreLayer(
        fuse_pool3, upshape=tf.shape(logits['images']),
        num_classes=num_classes, name='upscore32', ksize=16, stride=8)

    decoded_logits = {}
    decoded_logits['logits'] = upscore32
    decoded_logits['softmax'] = decoder.addSoftmax(hypes, upscore32)

    return decoded_logits
    

class ExpoSmoother():    
    def __init__(self, decay=0.9):
        self.weights = None
        self.decay = decay

    def update_weights(self, l):
        if self.weights is None:
            self.weights = np.array(l)
            return self.weights
        else:
            self.weights = self.decay*self.weights + (1-self.decay)*np.array(l)
            return self.weights

    def get_weights(self):
        return self.weights.tolist()


# class MedianSmoother():    
#     def __init__(self, num_entries=50):
#         self.weights = None
#         self.num = 50

#     def update_weights(self, l):
#         l = np.array(l).tolist()
#         if self.weights is None:
#             self.weights = [[i] for i in l]
#             return [np.median(w[-self.num:]) for w in self.weights]
#         else:
#             for i, w in enumerate(self.weights):
#                 w.append(l[i])
#             if len(self.weights) > 20*self.num:
#                 self.weights = [w[-self.num:] for w in self.weights]
#             return [np.median(w[-self.num:]) for w in self.weights]

#     def get_weights(self):
#         return [np.median(w[-self.num:]) for w in self.weights]

def loss(hypes, decoded_logits, labels):
    logits = decoded_logits['logits']
    num_classes = hypes['arch']['num_classes']

    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        shape = [logits.get_shape()[0], num_classes]
        epsilon = tf.constant(value=hypes['solver']['epsilon'])        
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
        softmax = tf.nn.softmax(logits) + epsilon
        cross_entropy_mean = computXEntropy(hypes, labels, softmax)

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
        weight_loss = tf.add_n(tf.get_collection(reg_loss_col), name='reg_loss')
        total_loss = cross_entropy_mean + weight_loss

        losses = {}
        losses['total_loss'] = total_loss
        losses['xentropy'] = cross_entropy_mean
        losses['weight_loss'] = weight_loss

    return losses


def computXEntropy(hypes, labels, softmax):
    head = hypes['arch']['weight']
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), head), reduction_indices=[1])
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return cross_entropy_mean


def evaluation(hypes, images, labels, decoded_logits, losses, global_step):
    eval_list = []
    num_classes = hypes['arch']['num_classes']
    logits = tf.reshape(decoded_logits['logits'], (-1, num_classes))
    labels = tf.reshape(labels, (-1, num_classes))
    pred = tf.argmax(logits, axis=1)
    y = tf.argmax(labels, 1)
    Prec = []
    Rec = []
    f1 = []

    for i in range(num_classes):
        tp = tf.count_nonzero(tf.cast(tf.equal(pred,i),tf.int32) * tf.cast(tf.equal(y,i),tf.int32))
        tn = tf.count_nonzero(tf.cast(tf.not_equal(pred,i),tf.int32) * tf.cast(tf.not_equal(y,i),tf.int32))
        fp = tf.count_nonzero(tf.cast(tf.equal(pred,i),tf.int32) * tf.cast(tf.not_equal(y,i),tf.int32))
        fn = tf.count_nonzero(tf.cast(tf.not_equal(pred,i),tf.int32) * tf.cast(tf.equal(pred,i),tf.int32))
        
        aux1 = tp + fp
        if(aux1 != 0):
            Prec.append(tp / aux1)
        else:
            Prec.append(tp)

        aux2 = tp + fn
        if(aux2 != 0):
            Rec.append(tp / aux2)
        else:
            Rec.append(tp)

        f1.append((2 * Prec[-1] * Rec[-1]) / (Prec[-1] + Rec[-1]))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, pred), tf.float32))

    # tf.summary.scalar("Accuracy", accuracy)
    # tf.summary.scalar("c1_Precision", Prec[1])
    # tf.summary.scalar("c1_Recall", Rec[1])
    #tf.summary.scalar("c1_F1_Score", f1[1])
    # tf.summary.scalar("c2_Precision", Prec[2])
    # tf.summary.scalar("c2_Recall", Rec[2])
    #tf.summary.scalar("c2_F1_Score", f1[2])
    # tf.summary.scalar("c3_Precision", Prec[3])
    # tf.summary.scalar("c3_Recall", Rec[3])
    #tf.summary.scalar("c3_F1_Score", f1[3])

    eval_list.append(('Acc. ', accuracy))
    eval_list.append(('xEntropy', losses['xentropy']))
    eval_list.append(('weight_loss', losses['weight_loss']))

    # Prec = tf.convert_to_tensor(Prec)
    # Rec = tf.convert_to_tensor(Rec)
    f1 = tf.convert_to_tensor(f1)
    # eval_list.append(('Overall Precision ', tf.reduce_mean(Prec)))
    # eval_list.append(('Overall Recall', tf.reduce_mean(Rec)))
    eval_list.append(('Overall F1 score ', tf.reduce_mean(f1)))

    return eval_list