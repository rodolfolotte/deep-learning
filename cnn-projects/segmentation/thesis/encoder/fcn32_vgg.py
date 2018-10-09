"""
Utilize vgg_fcn8 as encoder.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import tensorflow as tf

from tensorflow_fcn import fcn32_vgg

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

def inference(hypes, images, train=True):    
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """

    logging.info("....Creating encoder inference")

    vgg16_npy_path = os.path.join(hypes['dirs']['cnn_dir'], 'weights', "vgg16.npy")
    vgg_fcn = fcn32_vgg.FCN32VGG(vgg16_npy_path=vgg16_npy_path)
    vgg_fcn.wd = hypes['wd']
    
    logging.info(".....Number of classes: " + str(hypes['arch']['num_classes']))    
    vgg_fcn.build(images, train=train, num_classes=hypes['arch']['num_classes'], random_init_fc8=True)

    logits = {}
    logits['images'] = images

    if hypes['arch']['fcn_in'] == 'pool5':
        logits['fcn_in'] = vgg_fcn.pool5
    elif hypes['arch']['fcn_in'] == 'fc7':
        logits['fcn_in'] = vgg_fcn.fc7
    else:
        raise NotImplementedError

    logits['feed2'] = vgg_fcn.pool4
    logits['feed4'] = vgg_fcn.pool3

    #logits['fcn_logits'] = vgg_fcn.upscore64

    logits['deep_feat'] = vgg_fcn.pool5
    logits['early_feat'] = vgg_fcn.conv4_3

    return logits
