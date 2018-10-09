import itertools
import json
import logging
import os
import sys
import random
import threading
import numpy as np
import scipy as scp
import scipy.misc

import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes

from random import shuffle

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def prepareGT(hypes, data_file):
    base_path = os.path.realpath(os.path.dirname(data_file))
    files = [line.rstrip() for line in open(data_file)]

    for epoche in itertools.count():
        shuffle(files)
        for file in files:
            image_file, gt_image_file = file.split(" ")
            image_file = os.path.join(base_path, image_file)
            assert os.path.exists(image_file), "File does not exist: %s" % image_file
            gt_image_file = os.path.join(base_path, gt_image_file)
            assert os.path.exists(gt_image_file), "File does not exist: %s" % gt_image_file
            image = scipy.misc.imread(image_file, mode='RGB')
            gt_image = scp.misc.imread(gt_image_file, mode='RGB')

            yield image, gt_image


def prepareInputs(hypes, phase, data_dir):
   if phase == 'train':
       data_file = hypes['data']["train_file"]
   elif phase == 'val':
       data_file = hypes['data']["val_file"]
  
   data_file = os.path.join(data_dir, data_file)

   classes = hypes['colors']
   num_classes = len(classes)
   if(num_classes <= 2):
       print("Min amount of segmentation classes is 2 but only " + str(num_classes) + " class(es) is defined")
  
   data = prepareGT(hypes, data_file)

   for image, gt_image in data:
       gt_classes = []
       for color in classes.values():
           gt_classes.append(np.all(gt_image == color, axis=2))
       assert(gt_classes[0].shape == gt_classes[-1].shape)

       gt_reshaped = []
       shape = gt_classes[0].shape
       for gt_class in gt_classes:
           gt_reshaped.append(gt_class.reshape(shape[0], shape[1], 1))
        
       gt_image = np.concatenate(gt_reshaped, axis=2)
    
       if phase == 'val':
            yield image, gt_image
       elif phase == 'train':
            yield jitter_input(hypes, image, gt_image)
            yield jitter_input(hypes, np.fliplr(image), np.fliplr(gt_image))


def jitter_input(hypes, image, gt_image):
    jitter = hypes['jitter']
    res_chance = jitter['res_chance']
    crop_chance = jitter['crop_chance']

    if jitter['random_resize'] and res_chance > random.random():
        lower_size = jitter['lower_size']
        upper_size = jitter['upper_size']
        sig = jitter['sig']
        image, gt_image = random_resize(image, gt_image, lower_size, upper_size, sig)
        image, gt_image = crop_to_size(hypes, image, gt_image)

    if jitter['random_crop'] and crop_chance > random.random():
        max_crop = jitter['max_crop']
        crop_chance = jitter['crop_chance']
        image, gt_image = random_crop_soft(image, gt_image, max_crop)

    if jitter['reseize_image']:
        image_height = jitter['image_height']
        image_width = jitter['image_width']
        image, gt_image = resize_label_image(image, gt_image, image_height, image_width)

    if jitter['crop_patch']:
        patch_height = jitter['patch_height']
        patch_width = jitter['patch_width']
        image, gt_image = random_crop(image, gt_image, patch_height, patch_width)

    assert(image.shape[:-1] == gt_image.shape[:-1])
    return image, gt_image


def random_crop_soft(image, gt_image, max_crop):
    offset_x = random.randint(1, max_crop)
    offset_y = random.randint(1, max_crop)

    if random.random() > 0.5:
        image = image[offset_x:, offset_y:, :]
        gt_image = gt_image[offset_x:, offset_y:, :]
    else:
        image = image[:-offset_x, :-offset_y, :]
        gt_image = gt_image[:-offset_x, :-offset_y, :]

    return image, gt_image


def createQueues(hypes, phase):    
    arch = hypes['arch']
    dtypes = [tf.float32, tf.int32]

    shape_known = hypes['jitter']['reseize_image'] or hypes['jitter']['crop_patch']

    if shape_known:
        if hypes['jitter']['crop_patch']:
            height = hypes['jitter']['patch_height']
            width = hypes['jitter']['patch_width']
        else:
            height = hypes['jitter']['image_height']
            width = hypes['jitter']['image_width']

        channel = hypes['arch']['num_channels']
        num_classes = hypes['arch']['num_classes']
        shapes = [[height, width, channel],
                  [height, width, num_classes]]
    else:
        shapes = None

    capacity = 50
    q = tf.FIFOQueue(capacity=capacity, dtypes=dtypes, shapes=None)
    tf.summary.scalar("queue/%s/fraction_of_%d_full" %
                      (q.name + "_" + phase, capacity),
                      math_ops.cast(q.size(), tf.float32) * (1. / capacity))

    return q


def startEnqueuingThreads(hypes, q, phase, sess):    
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.int32)
    data_dir = hypes['data']['data_dir']

    def make_feed(data):
        image, label = data
        return {image_pl: image, label_pl: label}

    def enqueue_loop(sess, enqueue_op, phase, gen):        
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    enqueue_op = q.enqueue((image_pl, label_pl))
    gen = prepareInputs(hypes, phase, data_dir)
    gen.next()
    
    if phase == 'val':
        num_threads = 1
    else:
        num_threads = 1

    for i in range(num_threads):
        t = threading.Thread(target=enqueue_loop, args=(sess, enqueue_op, phase, gen))
        t.daemon = True
        t.start()


def _processe_image(hypes, image):
    augment_level = hypes['jitter']['augment_level']
    if augment_level > 0:
        image = tf.image.random_brightness(image, max_delta=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    if augment_level > 1:
        image = tf.image.random_hue(image, max_delta=0.15)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)

    return image


def inputs(hypes, q, phase):    
    if phase == 'val':
        image, label = q.dequeue()
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
        return image, label

    shape_known = hypes['jitter']['reseize_image'] or hypes['jitter']['crop_patch']

    if not shape_known:
        image, label = q.dequeue()
        nc = hypes["arch"]["num_classes"]
        label.set_shape([None, None, nc])
        image.set_shape([None, None, 3])
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
        if hypes['solver']['batch_size'] > 1:
            logging.error("Using a batch_size of {} with unknown shape."
                          .format(hypes['solver']['batch_size']))
            logging.error("Set batch_size to 1 or use `reseize_image` "
                          "or `crop_patch` to obtain a defined shape")
            raise ValueError
    else:
        image, label = q.dequeue_many(hypes['solver']['batch_size'])

    image = _processe_image(hypes, image)

    tensor_name = image.op.name
    tf.summary.image(tensor_name + '/image', image)

    facade = tf.expand_dims(tf.to_float(label[:, :, :, 0]), 3)
    tf.summary.image(tensor_name + '/gt_image', facade)

    return image, label
