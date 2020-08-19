# A improvement of tfrecord file generation presented by Daniil in:
#	https://github.com/warmspringwinds/tensorflow_notes/blob/master/tfrecords_guide.ipynb

import numpy as np
import skimage.io as io
import os
import sys
import tensorflow as tf

from PIL import Image
from os.path import basename

originalimages_full = sys.argv[1]
annotation_full = sys.argv[2]
tfrecords_filename = sys.argv[3] + ".tfrecords"

filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=10)

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def construct_pairs(images_path, annotation_path):
	pairs = []

	print("Constructing the pairs of images and its respectively annotations...")

	for file in os.listdir(images_path):
		if(file.endswith(".jpg")):
			absolute_image_name = os.path.join(images_path, file)
			image_name = basename(absolute_image_name)

			name, file_extension = os.path.splitext(image_name)
			
			annotation_name = annotation_path + name + ".png"

			if(os.path.isfile(annotation_name)):
				pair = (absolute_image_name, annotation_name)
				pairs.append(pair)				

	return pairs


def image2tfrecords(file_pairs):
	tfrecords_pairs = []

	print("Building the tfrecords file...")

	for img_path, ann_path in file_pairs:
	    
	    img = np.array(Image.open(img_path))
	    annotation = np.array(Image.open(ann_path))
	    
	    height = img.shape[0]
	    width = img.shape[1]
	    
	    tfrecords_pairs.append((img, annotation))
	    
	    img_raw = img.tostring()
	    annotation_raw = annotation.tostring()
	    
	    example = tf.train.Example(features=tf.train.Features(feature={
	        'height': _int64_feature(height),
	        'width': _int64_feature(width),
	        'image_raw': _bytes_feature(img_raw),
	        'mask_raw': _bytes_feature(annotation_raw)}))
	    
	    writer.write(example.SerializeToString())

	writer.close()

	return tfrecords_pairs


def tfrecords2image():
	reconstructed_images = []
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

	print("Reconstructing the images with tfrecords file...")

	for string_record in record_iterator:
	    
	    example = tf.train.Example()
	    example.ParseFromString(string_record)
	    
	    height = int(example.features.feature['height']
	                                 .int64_list
	                                 .value[0])
	    
	    width = int(example.features.feature['width']
	                                .int64_list
	                                .value[0])
	    
	    img_string = (example.features.feature['image_raw']
	                                  .bytes_list
	                                  .value[0])
	    
	    annotation_string = (example.features.feature['mask_raw']
	                                .bytes_list
	                                .value[0])
	    
	    
	    img_1d = np.fromstring(img_string, dtype=np.uint8)

	    reconstructed_img = img_1d.reshape((height, width, -1))	    
	    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
	    
	    reconstructed_annotation = annotation_1d.reshape((height, width))	    
	    reconstructed_images.append((reconstructed_img, reconstructed_annotation))

	return reconstructed_images


def check_match(original_images_path, reconstructed_images_path):
	print("Checking if the reconstructed images matches with original ones...")

	for original_pair, reconstructed_pair in zip(original_images_path, reconstructed_images_path):
	    
	    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair, reconstructed_pair)

	    if(not np.allclose(*img_pair_to_compare)):	    	
	    	print(">> The original raster " + original_pair + " is not the same as the reconstructed: " + reconstructed_pair + " !")

		if(not np.allclose(*annotation_pair_to_compare)):			
			print(">> The annotated raster " + original_pair + " is not the same as the reconstructed: " + reconstructed_pair + " !")


if __name__ == "__main__":

	original_images = []
	reconstructed_images = []
	file_pairs = construct_pairs(originalimages_full, annotation_full)
	
	original_images = image2tfrecords(file_pairs)
	reconstructed_images = tfrecords2image()
	
	check_match(original_images, reconstructed_images)