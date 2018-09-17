import numpy as np
import sys, os
import skimage.io as io

from os.path import basename
from random import shuffle

# EXAMPLE:
# python prepare-input.py PATH/dataset/ PATH/annotation/ PATH/ruemonge2014/inputs/ PERCENTAGE
def create_test_list(image_folder, annotation_folder, output_test_list):

    # Make pairs with only images that have its respective annotation
    with open(output_test_list, 'w+') as output:
        for filename in os.listdir(image_folder):
            absolute_image_name = os.path.join(image_folder, filename)
            image_name = basename(absolute_image_name)

            name, file_extension = os.path.splitext(image_name)

            annotation_name = annotation_folder + name + file_extension

            if(not os.path.isfile(annotation_name)):
                output.write(image_folder + filename + '\n')

def create_file_list(image_folder, annotation_folder, output_file_list):

    # Make pairs with only images that have its respective annotation
    with open(output_file_list, 'w+') as output:
        for filename in os.listdir(image_folder):
            absolute_image_name = os.path.join(image_folder, filename)
            image_name = basename(absolute_image_name)

            name, file_extension = os.path.splitext(image_name)

            annotation_name = annotation_folder + name + file_extension

            if(os.path.isfile(annotation_name)):
                output.write(image_folder + filename + ' ' + annotation_name + '\n')


if __name__ == '__main__':

    all_file = "all.txt"
    train_file = "train.txt"
    val_file = "val.txt"
    test_file = "tests.txt"

    image_folder = sys.argv[1]
    annotation_folder = sys.argv[2]
    output = sys.argv[3]
    percentage_val = float(sys.argv[4])

    file_list = os.path.join(output, all_file)
    train_file = os.path.join(output, train_file)
    val_file = os.path.join(output, val_file)
    test_file = os.path.join(output, test_file)

    if not os.path.exists(image_folder):
        print("Image folder not found: %s", image_folder)
        exit(1)

    if not os.path.exists(annotation_folder):
        print("Annotation folder not found: %s", annotation_folder)
        exit(1)

    if not os.path.exists(file_list):
        create_file_list(image_folder, annotation_folder, file_list)

    if not os.path.exists(test_file):
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