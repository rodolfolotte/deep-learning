import os
import sys
import glob
import logging
import time
import argparse
import settings
import sm_dataset as dataset
import sm_model as model

from coloredlogs import ColoredFormatter

sys.setrecursionlimit(1500)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)
logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

logging.getLogger("boto3").propagate = False
logging.getLogger("botocore").propagate = False


def main(arguments):
    """
    USAGE: python main.py -verbose True

    :param arguments: String with string args
    :return:
    """
    start_time = time.time()
    logging.info("Starting process...")

    train_path = os.path.join(settings.STRUCTURED_DATASET_DIR, 'train')
    validation_path = os.path.join(settings.STRUCTURED_DATASET_DIR, 'validation')
    train_annotation_path = os.path.join(settings.STRUCTURED_DATASET_DIR, 'train_annotation')
    validation_annotation_path = os.path.join(settings.STRUCTURED_DATASET_DIR, 'validation_annotation')

    dataset_obj = dataset.Dataset(train_path, validation_path,
                                  train_annotation_path, validation_annotation_path,
                                  'semantic-segmentation-demo')

    if (arguments.prepare_local_dir is not None) and (eval(arguments.prepare_local_dir)):
        dataset_obj.prepare_default_dir_structure('VOCdevkit/VOC2012', 'ImageSets/Segmentation/train.txt',
                                                  'ImageSets/Segmentation/val.txt',
                                                  'JPEGImages',
                                                  'SegmentationClass')

    if (arguments.transfer_dir_to_s3 is not None) and (eval(arguments.transfer_dir_to_s3)):
        dataset_obj.prepare_dirs_in_s3()

    model_obj = model.Training()
    if (arguments.training is not None) and (eval(arguments.training)):
        model_obj.get_docker_dl_image(dataset_obj.sess)
        model_obj.get_estimator(dataset_obj.sess, settings.PERSONAL_AWS_ROLE, dataset_obj.output)
        model_obj.setup_hyperparameter(len(glob.glob1(dataset_obj.train_path, "*.jpg")))
        model_obj.train(dataset_obj.sess, dataset_obj.bucket, dataset_obj.train_channel, dataset_obj.validation_channel,
                        dataset_obj.train_annotation_channel, dataset_obj.validation_annotation_channel)

    """USAGE: python main.py -endpoint s3://sagemaker-us-east-1-493849984591/semantic-segmentation-demo/
              output/ss-notebook-demo-2020-06-27-04-28-25-905/output/model.tar.gz"""
    if arguments.endpoint is not None:
        model_obj.get_docker_dl_image(dataset_obj.sess)
        model_obj.create_endpoint_from_a_model_in_s3(dataset_obj.sess, settings.PERSONAL_AWS_ROLE, arguments.endpoint)

    if (arguments.inference is not None) and (eval(arguments.inference)):
        for file in os.listdir(settings.INFERENCE_FOLDER):
            if file.endswith(".jpg"):
                model_obj.infer(os.path.join(settings.INFERENCE_FOLDER, file), settings.INFERENCE_OUTPUT,
                                settings.ENDPOINT, False)

    end_time = time.time()
    logging.info("Whole process completed! [Time: {0:.5f} seconds]!".format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DESCRIPTION')
    parser.add_argument('-prepare_local_dir', action="store", dest='prepare_local_dir',
                        help='Boolean to prepare an unorganized dataset, in a standard and well organized structure')
    parser.add_argument('-transfer_dir_to_s3', action="store", dest='transfer_dir_to_s3',
                        help='Boolean to transfer local training dataset, to S3 bucket')
    parser.add_argument('-training', action="store", dest='training', help='Boolean for training or not the DL model')
    parser.add_argument('-endpoint', action="store", dest='endpoint', help='String with a complete S3 model path')
    parser.add_argument('-inference', action="store", dest='inference',
                        help='Boolean for infering over a specific folder, to be specified in settings.py')
    parser.add_argument('-verbose', action="store", dest='verbose', help='Print log of processing')
    args = parser.parse_args()

    if eval(args.verbose):
        log = logging.getLogger('')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        cf = ColoredFormatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ")
        ch.setFormatter(cf)
        log.addHandler(ch)

        fh = logging.FileHandler('logging.log')
        fh.setLevel(logging.INFO)
        ff = logging.Formatter("[%(asctime)s] {%(filename)-15s:%(lineno)-4s} %(levelname)-5s: %(message)s ",
                               datefmt='%Y.%m.%d %H:%M:%S')
        fh.setFormatter(ff)
        log.addHandler(fh)

        log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")

    main(args)
