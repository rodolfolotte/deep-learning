import os
import glob
import json
import logging
import sagemaker
import shutil
import settings


class Dataset:
    """ """

    def __init__(self, train_path, validation_path, train_ann_path, validation_ann_path, prefix):
        self.sess = sagemaker.Session()
        self.bucket = self.sess.default_bucket()
        self.prefix = prefix
        self.output = 's3://{}/{}/output'.format(self.bucket, self.prefix)

        self.train_channel = os.path.join(self.prefix, 'train')
        self.validation_channel = os.path.join(self.prefix, 'validation')
        self.train_annotation_channel = os.path.join(self.prefix, 'train_annotation')
        self.validation_annotation_channel = os.path.join(self.prefix, 'validation_annotation')

        self.train_path = train_path
        self.validation_path = validation_path
        self.train_annotation_path = train_ann_path
        self.validation_annotation_path = validation_ann_path

    def prepare_dirs_in_s3(self):
        """"""
        self.sess.upload_data(path=self.train_path, bucket=self.bucket, key_prefix=self.train_channel)
        self.sess.upload_data(path=self.validation_path, bucket=self.bucket, key_prefix=self.validation_channel)
        self.sess.upload_data(path=self.train_annotation_path, bucket=self.bucket,
                              key_prefix=self.train_annotation_channel)
        self.sess.upload_data(path=self.validation_annotation_path, bucket=self.bucket,
                              key_prefix=self.validation_annotation_channel)

    def create_txt_list_image(self, dataset_path, train_txt_path, val_txt_path):
        """"""
        pascoal_voc_path = os.path.join(settings.DATASET_BASE_PATH, dataset_path)
        filename = os.path.join(pascoal_voc_path, train_txt_path)

        with open(filename) as f:
            train_list = f.read().splitlines()

        filename = os.path.join(pascoal_voc_path, val_txt_path)
        with open(filename) as f:
            val_list = f.read().splitlines()

        label_map = {"scale": 1}
        with open(settings.STRUCTURED_DATASET_DIR + '/train_label_map.json', 'w') as lm_fname:
            json.dump(label_map, lm_fname)

        return train_list, val_list

    def prepare_default_dir_structure(self, dataset_path, train_txt_file, val_txt_file, images_folder, labels_folder):
        """"""
        train_list, val_list = self.create_txt_list_image(dataset_path, train_txt_file, val_txt_file)
        pascoal_voc_path = os.path.join(settings.DATASET_BASE_PATH, dataset_path)

        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.validation_path, exist_ok=True)
        os.makedirs(self.train_annotation_path, exist_ok=True)
        os.makedirs(self.validation_annotation_path, exist_ok=True)

        for i in train_list:
            images_path = os.path.join(pascoal_voc_path, images_folder)
            labels_path = os.path.join(pascoal_voc_path, labels_folder)

            shutil.copy2(images_path + "/" + i + '.jpg', self.train_path)
            shutil.copy2(labels_path + "/" + i + '.png', self.train_annotation_path)

        for i in val_list:
            shutil.copy2(images_path + "/" + i + '.jpg', self.validation_path)
            shutil.copy2(labels_path + "/" + i + '.png', self.validation_annotation_path)

        num_training_samples = len(glob.glob1(self.train_path, "*.jpg"))
        num_training_ann_samples = len(glob.glob1(self.train_annotation_path, "*.png"))
        num_validation_samples = len(glob.glob1(self.validation_path, "*.jpg"))
        num_validation_ann_samples = len(glob.glob1(self.validation_annotation_path, "*.png"))

        logging.info('>> Num Train Images = ' + str(num_training_samples))
        assert num_training_samples == num_training_ann_samples

        logging.info('>> Num Validation Images = ' + str(num_validation_samples))
        assert num_validation_samples == num_validation_ann_samples



