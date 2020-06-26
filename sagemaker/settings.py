from decouple import config

DATASET_BASE_PATH = config('DATASET_BASE_PATH', default='/data/lotte/pascalvoc/original')
STRUCTURED_DATASET_DIR = config('STRUCTURED_DATASET_DIR', default='/data/lotte/pascalvoc/modified')

TRAINING_AWS_INSTANCE = 'ml.p3.2xlarge'
PERSONAL_AWS_ROLE = 'arn:aws:iam::493849984591:role/AmazonSageMaker'
ENDPOINT = 'ss-notebook-demo-2020-06-19-18-14-21-048'

"""
    backbone: This is the encoder. The options are 'resnet-50' and 'resnet-101'
    algorithm: This is the decoder. The options are 'psp', 'fcn' and 'deeplab'
    use_pretrained_model: Use the pre-trained model
    crop_size: Size of image random crop
    num_classes: Pascal has 21 classes. This is a mandatory parameter
    epochs: Number of epochs to run
    learning_rate:
    optimizer: The options are 'adam', 'rmsprop', 'nag', 'adagrad'
    lr_scheduler: The options are 'poly', 'cosine' and 'step'
    mini_batch_size: Setup some mini batch size
    validation_mini_batch_size: 
    early_stopping: Turn on early stopping. If OFF, other early stopping parameters are ignored
    early_stopping_patience: Tolerate these many epochs if the mIoU doens't increase
    early_stopping_min_epochs: No matter what, run these many number of epochs
    num_training_samples: This is a mandatory parameter
"""
HYPER = {'model': 'resnet-50', 'algorithm': 'fcn', 'use_pretrained_model': True, 'crop_size': 240,
         'num_classes': 21, 'epochs': 10, 'learning_rate': 0.0001, 'optmizer': 'rmsprop',
         'lr_scheduler': 'poly', 'mini_batch_size': 16, 'validation_mini_batch_size': 16,
         'early_stopping': True, 'early_stopping_patience': 2, 'early_stopping_min_epochs': 10}
