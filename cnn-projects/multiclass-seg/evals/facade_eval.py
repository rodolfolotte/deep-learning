"""
Evaluate CNN models for facade features detection 

Utilizes: Trained FacadeSeg weights. If no logdir is given, fails.

--------------------------------------------------------------------------------
This is an extension code from KittSeg

Author: MarvinTeichmann

Link: https://github.com/MarvinTeichmann/KittiSeg
--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Rodolfo Lotte

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import logging 
import time
import numpy as np
import scipy as scp

import tensorflow as tf

import tensorvision
import tensorvision.utils as utils

from evaluation import seg_utils as seg

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)


# Bendidi
#def eval_image(hypes, gt_image, cnn_image):
#    """."""
#    thresh = np.array(range(0, 256))/255.0
#    FN,FP = np.zeros(thresh.shape),np.zeros(thresh.shape)
#    posNum, negNum = 0,0
#
#    bg = np.array(hypes['colors']['background'])
#    sky = np.array(hypes['colors']['sky'])
#    roof = np.array(hypes['colors']['roof'])
#    wall = np.array(hypes['colors']['wall'])
#    window = np.array(hypes['colors']['wind#ow'])
#    door = np.array(hypes['colors']['door'])
#    shop = np.array(hypes['colors']['shop'])
#    balcony = np.array(hypes['colors']['balcony'])

#    gt_bg = np.all(gt_image == bg, axis=2)
#    gt_sky = np.all(gt_image == sky, axis=2)
#    gt_roof = np.all(gt_image == roof, axis=2)
#    gt_wall = np.all(gt_image == wall, axis=2)
#    gt_window = np.all(gt_image == window, axis=2)
#    gt_door = np.all(gt_image == door, axis=2)
#    gt_shop = np.all(gt_image == shop, axis=2)
#    gt_balcony = np.all(gt_image == balcony, axis=2)

#    valid_gt = gt_bg + gt_sky + gt_roof + gt_wall + gt_window + gt_door + gt_shop + gt_balcony
#    colors = [gt_sky,gt_roof,gt_wall,gt_window,gt_door,gt_shop,gt_balcony]

#    for i in range(len(colors)) :
#        N, P, pos, neg = seg.evalExp(np.all(gt_image == colors[i], axis=2),
#                                             cnn_image,
#                                             thresh, validMap=None,
#                                             validArea=valid_gt)
#        FN = np.add(FN,N)
#        FP = np.add(FP,P)
#        posNum+=pos
#        negNum+=neg

#    return FN, FP, posNum, negNum


def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0
    FN, FP = np.zeros(thresh.shape),np.zeros(thresh.shape)
    posNum, negNum = 0,0
    
    colors = []
    for key in hypes['colors']:
        colors.append(np.array(hypes['colors'][key]))
    
    # valid_gt = np.all(gt_image == colors[0], axis=2)
    valid_gt = 0
    for i in range(len(colors)) :
        valid_gt = valid_gt + np.all(gt_image == colors[i], axis=2)

    for i in range(len(colors)) :
        N, P, pos, neg = seg.evalExp(np.all(gt_image == colors[i], axis=2),
                                             cnn_image,
                                             thresh, validMap=None,
                                             validArea=valid_gt)
        FN=np.add(FN,N)
        FP=np.add(FP,P)
        posNum+=pos
        negNum+=neg

    return FN, FP, posNum, negNum



def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width), interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width), interp='nearest')

    return image, gt_image


def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['data']['data_dir']
    num_classes = hypes['arch']['num_classes']
    colors = hypes['colors']
    eval_dict = {}

    logging.info('Evaluating the ' + str(num_classes) + ' classes.')
    
    for phase in ['train', 'val']:
        data_file = hypes['data']['{}_file'.format(phase)]        
        data_file = os.path.join(data_dir, data_file)
        image_dir = os.path.dirname(data_file)

        thresh = np.array(range(0, 256))/255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        total_posnum = 0
        total_negnum = 0

        image_list = []

        with open(data_file) as file:
            for i, datum in enumerate(file):
                    datum = datum.rstrip()
                    image_file, gt_file = datum.split(" ")
                    image_file = os.path.join(image_dir, image_file)
                    gt_file = os.path.join(image_dir, gt_file)

                    image = scp.misc.imread(image_file, mode='RGB')
                    gt_image = scp.misc.imread(gt_file, mode='RGB')

                    if hypes['jitter']['fix_shape']:
                        shape = image.shape
                        image_height = hypes['jitter']['image_height']
                        image_width = hypes['jitter']['image_width']
                        assert(image_height >= shape[0])
                        assert(image_width >= shape[1])

                        offset_x = (image_height - shape[0])//2
                        offset_y = (image_width - shape[1])//2
                        new_image = np.zeros([image_height, image_width, 3])
                        new_image[offset_x:offset_x+shape[0],
                                  offset_y:offset_y+shape[1]] = image
                        input_image = new_image
                    elif hypes['jitter']['reseize_image']:
                        image_height = hypes['jitter']['image_height']
                        image_width = hypes['jitter']['image_width']
                        gt_image_old = gt_image
                        image, gt_image = resize_label_image(image, gt_image,
                                                             image_height,
                                                             image_width)
                        input_image = image
                    else:
                        input_image = image

                    shape = input_image.shape

                    feed_dict = {image_pl: input_image}

                    output = sess.run([softmax], feed_dict=feed_dict)
                    output_im = output[0].reshape(shape[0], shape[1], num_classes)
                    output_im = np.argmax(output_im,axis=2)

                    if hypes['jitter']['fix_shape']:
                        gt_shape = gt_image.shape
                        output_im = output_im[offset_x:offset_x+gt_shape[0], offset_y:offset_y+gt_shape[1]]

                    # if phase == 'val':
                    #     segmented_image = seg.paint(output_im, colors)
                    #     name = os.path.basename(image_file)
                    #     image_list.append((name, segmented_image))

                    #     name2 = name.split('.')[0] + '_overlay.png'

                    #     ov_image = seg.blend_transparent(image, segmented_image)
                    #     image_list.append((name2, ov_image))

                    FN, FP, posNum, negNum = eval_image(hypes, gt_image, output_im)

                    total_fp += FP
                    total_fn += FN
                    total_posnum += posNum
                    total_negnum += negNum

        eval_dict[phase] = seg.pxEval_maximizeFMeasure(total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)

        if phase == 'val':
            start_time = time.time()
            for i in xrange(10):
                sess.run([softmax], feed_dict=feed_dict)
            dt = (time.time() - start_time)/10

    eval_list = []

    for phase in ['train', 'val']:
        eval_list.append(('[{}] MaxF1'.format(phase),100*eval_dict[phase]['MaxF']))
        eval_list.append(('[{}] BestThresh'.format(phase),100*eval_dict[phase]['BestThresh']))
        eval_list.append(('[{}] Average Precision'.format(phase),100*eval_dict[phase]['AvgPrec']))

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list   
