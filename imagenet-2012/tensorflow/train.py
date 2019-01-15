import argparse
import time
import pickle

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import numpy as np
from models.alexnet_v2 import AlexNetV2
from models.resnet50 import ResNet50
from models.resnet152 import ResNet152
from data_load import preprocess_image

model_dir = './saved_models/'

training_config = {
    'alexnet2': {
        'name': 'alexnet2',
        'model': AlexNetV2,
        'batch_size': 128,
        'num_workers': 16,
        'optimizer': optimizers.SGD,
        # "...momentum may be less necessary...but in my experiments I used mu = 0.9..." alexnet2.[1]
        'optimizer_params': {
            'lr': 0.01,
            'momentum': 0.9,
        },
        'total_epochs': 200,
    },
    'resnet50': {
        'name': 'resnet50',
        'model': ResNet50,
        'batch_size': 128,
        'num_workers': 16,
        'optimizer': optimizers.SGD,
        # "...momentum may be less necessary...but in my experiments I used mu = 0.9..." alexnet2.[1]
        'optimizer_params': {
            'lr': 0.01,
            'momentum': 0.9,
        },
        'total_epochs': 200,
    },
    'resnet152': {
        'name': 'resnet152',
        'model': ResNet152,
        'batch_size': 128,
        'num_workers': 16,
        'optimizer': optimizers.SGD,
        # "...momentum may be less necessary...but in my experiments I used mu = 0.9..." alexnet2.[1]
        'optimizer_params': {
            'lr': 0.01,
            'momentum': 0.9,
        },
        'total_epochs': 200,
    }
}


class ModelHdf5Checkpoint(Callback):
    '''
    Save model as hdf5 format. The standard callback ModelCheckpoint saves in ckpt format
    '''

    def __init__(self, model_dir, model_filename, model_to_save):
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs={}):
        save_path = self.model_dir + self.model_filename + '-checkpoint-epoch-{}.hdf5'.format(
            epoch + 1)
        self.model_to_save.save(save_path)


class LoggersCallback(Callback):
    def __init__(self, path):
        self.path = path

    def on_train_begin(self, logs={}):
        self.loggers = {
            'train_loss': {
                'epochs': [],
                'value': [],
            },
            'train_top1_acc': {
                'epochs': [],
                'value': [],
            },
            'train_top5_acc': {
                'epochs': [],
                'value': [],
            },
            'val_loss': {
                'epochs': [],
                'value': [],
            },
            'val_top1_acc': {
                'epochs': [],
                'value': [],
            },
            'val_top5_acc': {
                'epochs': [],
                'value': [],
            },
            'lr': {
                'epochs': [],
                'value': [],
            },
        }

    def _log_metrics(self, name, value, epoch):
        logger = self.loggers.get(name)
        logger.get('epochs').append(epoch)
        logger.get('value').append(value)
        print('Epoch: {}, {}: {}'.format(
            epoch,
            name,
            value,
        ))

    def on_epoch_end(self, epoch, logs={}):
        real_epoch = epoch + 1
        lr = K.eval(self.model.optimizer.lr)
        self._log_metrics('train_loss', logs['loss'], real_epoch)
        self._log_metrics('train_top1_acc', logs['acc'], real_epoch)
        self._log_metrics('train_top5_acc', logs['top_5_accuracy'], real_epoch)
        self._log_metrics('val_loss', logs['val_loss'], real_epoch)
        self._log_metrics('val_top1_acc', logs['val_acc'], real_epoch)
        self._log_metrics('val_top5_acc', logs['val_top_5_accuracy'],
                          real_epoch)
        self._log_metrics('lr', lr, real_epoch)
        print('Time: {}'.format(
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())))
        with open(
                '{}-loggers-epoch-{}.pkl'.format(self.path, real_epoch),
                'wb',
        ) as f:
            pickle.dump(self.loggers, f, pickle.HIGHEST_PROTOCOL)


# https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_main.py#L62
def _parse_function(proto, is_training):
    # define your tfrecord again. Remember that you saved your image as a string.
    feature_map = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.FixedLenFeature([], tf.string),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/class/synset': tf.FixedLenFeature([], tf.string),
        'image/class/text': tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
    }

    features = tf.parse_single_example(proto, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    image = features['image/encoded']
    image = preprocess_image(
        image_buffer=image,
        output_height=224,
        output_width=224,
        num_channels=3,
        is_training=is_training,
    )
    image = tf.cast(image, tf.float32)

    return image, label


# https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
def create_dataset(filepath, config, is_training):

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(
        tf.data.Dataset.list_files(filepath),
        num_parallel_reads=config.get('num_workers'),
    )

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(
        lambda x: _parse_function(x, is_training),
        num_parallel_calls=config.get('num_workers'),
    )

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(10000)

    # Set the batchsize
    dataset = dataset.batch(config.get('batch_size'))

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    label = tf.one_hot(label, 1000)

    return image, label


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def run_epochs(config, checkpoint_path):

    steps_per_epoch = 1281167 // config.get('batch_size')
    validation_steps = 50000 // config.get('batch_size')

    train_image, train_label = create_dataset(
        '../dataset/tfrecord/tfrecord_train/*',
        config,
        True,
    )
    val_image, val_label = create_dataset(
        '../dataset/tfrecord/tfrecord_val/*',
        config,
        False,
    )

    # Create a the neural network
    Mdl = config.get('model')
    model_params = config.get('model_params')
    if model_params is not None:
        model = Mdl(input_shape=(224, 224, 3), **model_params)
    else:
        model = Mdl(input_shape=(224, 224, 3))
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path)

    devices = K.get_session().list_devices()
    model_to_use = model
    if 'GPU' in str(devices):
        gpus = len(devices)
        model_to_use = multi_gpu_model(model, gpus=gpus)

    # Define the optimizer
    Optim = config.get('optimizer')
    optimizer = Optim(**config.get('optimizer_params'))

    # Set model names
    model_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    model_name = config.get('name')
    model_filename = '{}-tf-{}'.format(model_name, model_id)

    # Define save checkpoint callback
    cp_callback = ModelHdf5Checkpoint(model_dir, model_filename, model)
    # Define save custom loggers callback
    lg_callback = LoggersCallback(model_dir + model_filename)

    # Define generate tensorboard log callback
    tb_callback = TensorBoard(
        log_dir='./tensorboard/{}'.format(model_filename))

    lr_callback = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)

    # Compile the model and generate computation graph
    model_to_use.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', top_5_accuracy],
    )
    model_to_use.summary()

    # Start training
    model_to_use.fit(
        train_image,
        train_label,
        epochs=config.get('total_epochs'),
        callbacks=[
            cp_callback,
            tb_callback,
            lg_callback,
            lr_callback,
        ],
        validation_data=(val_image, val_label),
        validation_steps=validation_steps,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    supported_models = list(training_config.keys())
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=supported_models,
        help="specify model name",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="specify checkpoint file path",
    )
    args = parser.parse_args()
    model_name = args.model
    checkpoint_path = args.checkpoint
    config = training_config.get(model_name)
    run_epochs(config, checkpoint_path)