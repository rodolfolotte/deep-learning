import os
import sagemaker
import PIL as pillow
import numpy as np
import matplotlib.pyplot as plt
import settings
import boto3
import mxnet as mx
import sm_utils as utils

from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.amazon.record_pb2 import Record


class Training:
    """"""
    def __init__(self):
        self.training_image = None
        self.estimator = None
        self.predictor = None

    def get_docker_dl_image(self, sess):
        """Get the docker image exclusively for semantic segmentation uses"""
        self.training_image = get_image_uri(sess.boto_region_name, 'semantic-segmentation', repo_version="latest")

    def get_estimator(self, sess, role, output):
        """
        Train the segmentation algorithm. Create a sageMaker.estimator.Estimator object. This estimator will launch
        the training job "ss-notebook-demo". Using a GPU instance (ml.p3.2xlarge) to train.

        :param sess:
        :param role:
        :param output:
        :return:
        """
        base_job_name = settings.HYPER['model'] + "-" + settings.HYPER['algorithm']

        self.estimator = sagemaker.estimator.Estimator(self.training_image,
                                                       role,
                                                       train_instance_count=1,
                                                       train_instance_type=settings.TRAINING_AWS_INSTANCE,
                                                       train_volume_size=50,
                                                       train_max_run=360000,
                                                       output_path=output,
                                                       base_job_name=base_job_name,
                                                       sagemaker_session=sess)

    def setup_hyperparameter(self, num_training_samples):
        """
        :param num_training_samples:
        :return:
        """
        self.estimator.set_hyperparameters(backbone=settings.HYPER['model'], algorithm=settings.HYPER['algorithm'],
                                           use_pretrained_model=settings.HYPER['use_pretrained_model'],
                                           crop_size=settings.HYPER['crop_size'],
                                           num_classes=settings.HYPER['num_classes'], epochs=settings.HYPER['epochs'],
                                           learning_rate=settings.HYPER['learning_rate'],
                                           optimizer=settings.HYPER['optmizer'],
                                           lr_scheduler=settings.HYPER['lr_scheduler'],
                                           mini_batch_size=settings.HYPER['mini_batch_size'],
                                           validation_mini_batch_size=settings.HYPER['validation_mini_batch_size'],
                                           early_stopping=settings.HYPER['early_stopping'],
                                           early_stopping_patience=settings.HYPER['early_stopping_patience'],
                                           early_stopping_min_epochs=settings.HYPER['early_stopping_min_epochs'],
                                           num_training_samples=num_training_samples)

    def train(self, sess, bucket, train_channel, validation_channel, train_annotation_channel, validation_annotation_channel):
        """
        :param sess:
        :param bucket:
        :param train_channel:
        :param validation_channel:
        :param train_annotation_channel:
        :param validation_annotation_channel:
        :return:
        """
        distribution = 'FullyReplicated'

        s3_train_data = 's3://{}/{}'.format(bucket, train_channel)
        s3_validation_data = 's3://{}/{}'.format(bucket, validation_channel)
        s3_train_annotation = 's3://{}/{}'.format(bucket, train_annotation_channel)
        s3_validation_annotation = 's3://{}/{}'.format(bucket, validation_annotation_channel)

        train_data = sagemaker.session.s3_input(s3_train_data, distribution=distribution,
                                                content_type='image/jpeg', s3_data_type='S3Prefix')
        validation_data = sagemaker.session.s3_input(s3_validation_data, distribution=distribution,
                                                     content_type='image/jpeg', s3_data_type='S3Prefix')
        train_annotation = sagemaker.session.s3_input(s3_train_annotation, distribution=distribution,
                                                      content_type='image/png', s3_data_type='S3Prefix')
        validation_annotation = sagemaker.session.s3_input(s3_validation_annotation, distribution=distribution,
                                                           content_type='image/png', s3_data_type='S3Prefix')

        data_channels = {'train': train_data,
                         'validation': validation_data,
                         'train_annotation': train_annotation,
                         'validation_annotation': validation_annotation}

        self.estimator.fit(inputs=data_channels, logs=True)
        self.predictor = self.create_endpoint(self.estimator)

    def infer(self, filename, output_folder, endpoint, show_image):
        """
        :param filename:
        :param output_folder:
        :param endpoint:
        :param show_image:
        :return:
        """
        output_filename = os.path.splitext(filename)[0]
        output_filename = os.path.basename(output_filename)
        output_filename = os.path.join(output_folder, output_filename + '_inference.png')

        runtime = boto3.Session().client('sagemaker-runtime')

        image = pillow.Image.open(filename)
        image.thumbnail([800, 600], pillow.Image.ANTIALIAS)
        image.save(filename, "JPEG")

        with open(filename, 'rb') as f:
            payload = f.read()
            payload = bytearray(payload)

        response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/x-image', Body=payload)
        results_file = 'results.rec'
        with open(results_file, 'wb') as f:
            f.write(response['Body'].read())

        rec = Record()
        recordio = mx.recordio.MXRecordIO(results_file, 'r')
        protobuf = rec.ParseFromString(recordio.read())
        values = list(rec.features["target"].float32_tensor.values)
        shape = list(rec.features["shape"].int32_tensor.values)
        shape = np.squeeze(shape)
        mask = np.reshape(np.array(values), shape)
        mask = np.squeeze(mask, axis=0)
        pred_map = np.argmax(mask, axis=0)

        if show_image is True:
            utils_obj = utils.Utils()
            utils_obj.show_image(pred_map)

        plt.imshow(pred_map, vmin=0, vmax=settings.HYPER['num_classes'] - 1, cmap='jet')
        plt.savefig(output_filename)

    def delete_endpoint(self, endpoint):
        """
        :param endpoint:
        :return:
        """
        sagemaker.Session().delete_endpoint(endpoint=endpoint)

    def create_endpoint(self, estimator):
        """
        :param estimator:
        :return:
        """
        predictor = estimator.deploy(initial_instance_count=1, instance_type=settings.TRAINING_AWS_INSTANCE)
        return predictor

    def create_endpoint_from_a_model_in_s3(self, sess, role, model_path_in_s3):
        """
        :param sess:
        :param role:
        :param model_path_in_s3:
        :return:
        """
        sm_model = sagemaker.Model(model_data=model_path_in_s3, image=self.training_image,
                                   role=role, sagemaker_session=sess)
        sm_model.deploy(initial_instance_count=1, instance_type=settings.TRAINING_AWS_INSTANCE)
