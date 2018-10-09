from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
import numpy as np
import scipy as scp
import scipy.misc
import collections
import tensorflow as tf

import dl.autoencoder as autoencoder
import inputs.prepare as inputs
import evaluate.eval as evaluation
import optimizer.optimizer as optimizer
import inference.inference as inference

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, stream=sys.stdout)  

# USAGE: python segment.py PATH_TO_THE_CHECKPOINT IMAGE_FOLDER OUTPUT_FOLDER

def loadWeights(checkpoint_dir, sess, saver):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:        
        file = os.path.basename(ckpt.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, file)
        saver.restore(sess, checkpoint_path)
        return int(file.split('-')[1])


def main(_):
    hypes_file = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]

    with open(hypes_file) as hypesFile:
        logging.info("Opening the neural model file settings (hyperparameters): %s", hypesFile)
        hypes = json.load(hypesFile)
 
    with tf.Graph().as_default():        
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)
        
        logging.info("Building inference graph...")
        prediction = inference.buildInferenceGraph(hypes, image=image)
        
        sess = tf.Session()
        saver = tf.train.Saver()
        
        logging.info("Loading weights (knownledge)...")        
        data_dict = loadWeights(hypes['data']['checkpoint_path'], sess, saver)

    classes_colors = []
    classes = hypes['colors']
    for color in classes.values():
        classes_colors.append(color)
    
    logging.info("Starting inferences under the {} folder...".format(input_folder))
    if(not os.listdir(input_folder)==""):
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                input_image = os.path.join(input_folder, file)

                logging.info(">> File {}...".format(file))                
                image = scp.misc.imread(input_image)
                           
                feed = {image_pl: image}
                softmax = prediction['softmax']
                logits = prediction['logits']
                output, lll = sess.run([softmax, logits], feed_dict=feed)  
                shape = image.shape
                output_image = output.reshape(shape[0], shape[1], -1)

                x = np.argmax(output_image, axis=2)
                im = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                for i,_ in enumerate(x):
                    for j,_ in enumerate(x[i]):
                        value = x[i][j]
                        color_code = classes_colors[value]
                        im[i][j] = color_code
               
                if(not os.path.isdir(output_folder)):
                    logging.info(">> Output directory does not exist. Creating in: " + output_folder)
                    os.makedirs(output_folder)
                
                raw_image_name = file.split('.')[0] + '.png'
                full_raw_image_name = os.path.join(output_folder, raw_image_name)

                scp.misc.imsave(full_raw_image_name, im)
                logging.info(">> Prediction saved: " + full_raw_image_name)   

    else:
        logging.info("Input folder is empty. Check it again")


if __name__ == '__main__':
    tf.app.run()
