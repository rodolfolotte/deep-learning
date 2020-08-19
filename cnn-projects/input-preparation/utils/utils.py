import os
import logging
import settings

from random import shuffle


class Utils:
    """ """

    def __init__(self):
        pass

    def check_file(self, dir, prefix):
        for s in os.listdir(dir):
            if os.path.splitext(s)[0] == prefix and os.path.isfile(os.path.join(dir, s)):
                return True

        return False

    def create_test_list(self, image_folder, annotation_folder, output_test_list):
        with open(output_test_list, 'w+') as output:
            for filename in os.listdir(image_folder):
                ext2 = os.path.splitext(filename)[1]

                if ext2.lower() not in settings.VALID_RASTER_EXTENSION:
                    continue

                absolute_image_name = os.path.join(image_folder, filename)
                image_name = os.path.basename(absolute_image_name)
                name, file_extension = os.path.splitext(image_name)

                if self.check_file(annotation_folder, name):
                    output.write(image_folder + filename + '\n')

    def create_file_list(self, image_folder, annotation_folder, output_file_list):
        with open(output_file_list, 'w+') as output:
            for filename in os.listdir(image_folder):
                ext2 = os.path.splitext(filename)[1]

                if ext2.lower() not in settings.VALID_RASTER_EXTENSION:
                    continue

                absolute_image_name = os.path.join(image_folder, filename)
                image_name = os.path.basename(absolute_image_name)

                name, file_extension = os.path.splitext(image_name)

                annotation_name = annotation_folder + name + desired_ann_ext

                if self.check_file(annotation_folder, name):
                    output.write(image_folder + filename + ' ' + annotation_name + '\n')

    def prepare_inputs(self, image_folder, annotation_folder, output, percentage_val):
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

        self.create_file_list(image_folder, annotation_folder, file_list)
        self.create_test_list(image_folder, annotation_folder, test_file)

        files = [line for line in open(file_list)]
        shuffle(files)

        percentage_val = int(round(len(files) * (percentage_val / 100)))

        train = files[:-percentage_val]
        val = files[-percentage_val:]

        with open(train_file, 'w+') as file:
            for label in train:
                file.write(label)

        with open(val_file, 'w+') as file:
            for label in val:
                file.write(label)

        logging.info(">> CNN inputs created successfully!")
