from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import random
import logging
import tensorflow as tf

from tensorflow.python.framework import dtypes
from math import ceil


def addSoftmax(hypes, logits):
    num_classes = hypes['arch']['num_classes']
    with tf.name_scope('decoder'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        
        softmax = tf.nn.softmax(logits)

    return softmax


def upscoreLayer(bottom, upshape, num_classes, name, ksize=4, stride=2):    
    logging.info('.......Layer ' +  name + ' ' + str(upshape))

    strides = [1, stride, stride, 1]

    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value
        new_shape = [upshape[0], upshape[1], upshape[2], num_classes]
        output_shape = tf.stack(new_shape)
        f_shape = [ksize, ksize, num_classes, in_features]
        up_init = upsampleInitilizer()
        weights = tf.get_variable(name="weights", initializer=up_init, shape=f_shape)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)

        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')
        
        # addToSummary(deconv)

    return deconv


def upsampleInitilizer(dtype=dtypes.float32):
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-float point type.')

    def _initializer(shape, dtype=dtype, partition_info=None):
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating type.')

        width = shape[0]
        heigh = shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([shape[0], shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(shape)

        for i in range(shape[2]):
            weights[:, :, i, i] = bilinear

        return weights

    return _initializer


# def addToSummary(x):
#     tensor_name = x.op.name
#     tf.summary.histogram(tensor_name + '/activations', x)
#     tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))