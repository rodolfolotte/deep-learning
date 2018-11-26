#!/usr/bin/env python

"""
Command-line routine that through two folders: original images and ground-truth, build validation, testing and training .txt files. Each line in these files represents pairs of paths: original image SPACE image reference. In the training file, all the pairs of images to be used in the training of the neural model are listed, while in the validation file, images are placed for validation according to the percentage specified
"""

__author__ = 'Rodolfo G. Lotte'
__copyright__ = 'Copyright 2018, Rodolfo G. Lotte'
__credits__ = ['Rodolfo G. Lotte']
__license__ = 'MIT'
__usage__ = 'python prepare-input.py -image_folder PATH/dataset/ -annotation_folder PATH/annotation/ -output_folder PATH/inputs/ -percentage PERCENTAGE'
__email__ = 'rodolfo.lotte@gmail.com'

import logging
import os
import sys
import argparse

from os.path import basename
from random import shuffle

valid_images = [".jpg",".gif",".png",".tga",".tif"]
desired_ann_ext = ".png"

log = logging.getLogger('')
log.setLevel(logging.INFO)
format = logging.Formatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ", datefmt='%Y.%m.%d %H:%M:%S')

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)

fh = logging.handlers.RotatingFileHandler(filename='inputs.log', maxBytes=(1048576*5), backupCount=7)
fh.setFormatter(format)

log.addHandler(ch)
log.addHandler(fh)

def check_file(dir, prefix):
    for s in os.listdir(dir):        
        if os.path.splitext(s)[0] == prefix and os.path.isfile(os.path.join(dir, s)):
            return True

    return False


def create_test_list(image_folder, annotation_folder, output_test_list):
    with open(output_test_list, 'w+') as output:
        for filename in os.listdir(image_folder):
            ext2 = os.path.splitext(filename)[1]
                
            if ext2.lower() not in valid_images:            
                continue

            absolute_image_name = os.path.join(image_folder, filename)
            image_name = basename(absolute_image_name)
            name, file_extension = os.path.splitext(image_name)

            if(check_file(annotation_folder, name)):                
                output.write(image_folder + filename + '\n')


def create_file_list(image_folder, annotation_folder, output_file_list):
    with open(output_file_list, 'w+') as output:
        for filename in os.listdir(image_folder):
            ext2 = os.path.splitext(filename)[1]
                
            if ext2.lower() not in valid_images:            
                continue

            absolute_image_name = os.path.join(image_folder, filename)
            image_name = basename(absolute_image_name)

            name, file_extension = os.path.splitext(image_name)

            annotation_name = annotation_folder + name + desired_ann_ext

            if(check_file(annotation_folder, name)):
                output.write(image_folder + filename + ' ' + annotation_name + '\n')


def prepareInputs(image_folder, annotation_folder, output, percentage_val):
    all_file = "all.txt"
    train_file = "train.txt"
    val_file = "val.txt"
    test_file = "tests.txt"

    file_list = os.path.join(output, all_file)
    train_file = os.path.join(output, train_file)
    val_file = os.path.join(output, val_file)
    test_file = os.path.join(output, test_file)

    if not os.path.exists(image_folder):
        logging.info(">> Image folder not found: %s", image_folder)
        exit(1)

    if not os.path.exists(annotation_folder):
        logging.info(">> Annotation folder not found: %s", annotation_folder)
        exit(1)

    # if not os.path.exists(file_list):
    create_file_list(image_folder, annotation_folder, file_list)

    # if not os.path.exists(test_file):    
    create_test_list(image_folder, annotation_folder, test_file)

    files = [line for line in open(file_list)]

    shuffle(files)

    percentage_val = int(round(len(files) * (percentage_val/100)))
    
    train = files[:-percentage_val]
    val = files[-percentage_val:]

    with open(train_file, 'w+') as file:
        for label in train:
            file.write(label)

    with open(val_file, 'w+') as file:
        for label in val:
            file.write(label)

    logging.info(">> CNN inputs created successfully!")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Prepare input files (txts) for supervised neural network procedures')
    
    parser.add_argument('-image_folder', action="store", dest='imageFolder', help='Images folder')            
    parser.add_argument('-annotation_folder', action="store", dest='annotationFolder', help='Annotations folder')
    parser.add_argument('-output_folder', action="store", dest='outputFolder', help='Folder to store the input files')
    parser.add_argument('-percentage', action="store", dest='percentage', help='Shapefiles folder, the folder with shapefiles references in order to be tiled')    
    
    result = parser.parse_args()

    logging.info("Preparing inputs...")
    prepareInputs(result.imageFolder, result.annotationFolder, result.outputFolder, result.percentage)