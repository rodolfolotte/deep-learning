import sagemaker
import settings

from sagemaker.amazon.amazon_estimator import get_image_uri


class Training:
    """"""
    def __init__(self):
        self.training_image = None
        self.estimator = None

    def get_docker_dl_image(self):
        """Get the docker image exclusively for semantic segmentation uses"""
        sess = sagemaker.Session()
        self.training_image = get_image_uri(sess.boto_region_name, 'semantic-segmentation', repo_version="latest")

    def get_estimator(self, sess, role, output):
        """
        Train the segmentation algorithm. Create a sageMaker.estimator.Estimator object. This estimator will launch
        the training job "ss-notebook-demo". Using a GPU instance (ml.p3.2xlarge) to train.
        """
        self.estimator = sagemaker.estimator.Estimator(self.training_image,
                                                       role,
                                                       train_instance_count=1,
                                                       train_instance_type=settings.TRAINING_AWS_INSTANCE,
                                                       train_volume_size=50,
                                                       train_max_run=360000,
                                                       output_path=output,
                                                       base_job_name='ss-notebook-demo',
                                                       sagemaker_session=sess)

    def setup_hyperparameter(self, num_training_samples):
        """
        :param estimator:
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

    def train(self, bucket, train_channel, validation_channel, train_annotation_channel, validation_annotation_channel):
        """ """
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
