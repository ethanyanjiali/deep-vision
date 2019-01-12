import argparse
import time
import pickle

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import mnist
import numpy as np
from models.alexnet_v2 import AlexNetV2
from data_load import preprocess_image

model_dir = './saved_models/'

training_config = {
    'alexnet2': {
        'name': 'alexnet2',
        'model': AlexNetV2,
        'batch_size': 128,
        'num_workers': 8,
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
    path: path to save hdf5 model file
    '''

    def __init__(self, model_dir, model_filename):
        self.model_dir = model_dir
        self.model_filename = model_filename

    def on_epoch_end(self, epoch, logs={}):
        save_path = self.model_dir + self.model_filename + '-checkpoint-epoch-{}.hdf5'.format(
            epoch + 1)
        self.model.save(save_path)


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
            'val_loss': {
                'epochs': [],
                'value': [],
            },
            'val_top1_acc': {
                'epochs': [],
                'value': [],
            },
        }

    def _log_metrics(self, name, value, epoch):
        logger = self.loggers.get(name)
        logger.get('epochs').append(epoch)
        logger.get('value').append(value)

    def on_epoch_end(self, epoch, logs={}):
        real_epoch = epoch + 1
        self._log_metrics('train_loss', logs['loss'], real_epoch)
        self._log_metrics('train_top1_acc', logs['acc'], real_epoch)
        self._log_metrics('val_loss', logs['val_loss'], real_epoch)
        self._log_metrics('val_top1_acc', logs['val_acc'], real_epoch)
        print('Time: {}'.format(
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())))
        print('Epoch: {}, Validation Top 1 acc: {}'.format(
            real_epoch,
            logs['val_acc'],
        ))
        print('Epoch: {}, Validation Set Loss: {}'.format(
            real_epoch,
            logs['val_loss'],
        ))
        with open(
                '{}-loggers-epoch-{}.pkl'.format(self.path, real_epoch),
                'wb',
        ) as f:
            pickle.dump(self.loggers, f, pickle.HIGHEST_PROTOCOL)


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    feature_map = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.FixedLenFeature([], tf.string),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64),
        'image/class/synset': tf.FixedLenFeature([], tf.string),
        'image/class/text': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/label': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
    }

    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update({
        k: sparse_float32
        for k in [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax'
        ]
    })

    features = tf.parse_single_example(example_serialized, feature_map)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    image = features['image/encoded']
    image = preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=224,
        output_width=224,
        num_channels=3,
        is_training=is_training,
    )
    image = tf.cast(image, dtype)

    return image, label


def create_dataset(filepath, config):

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(
        _parse_function, num_parallel_calls=config.get('num_workers'))

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

    return image, label


def run_epochs(config, checkpoint_path):

    steps_per_epoch = 1281167 // config.get('batch_size')

    train_image, train_label = create_dataset(
        '../dataset/tfrecord_train',
        config,
    )
    val_image, val_label = create_dataset(
        '../dataset/tfrecord_val',
        config,
    )

    # Create a the neural network
    Mdl = config.get('model')
    model_params = config.get('model_params')
    if model_params is not None:
        model = Mdl(input_shape=(224, 224, 3), **model_params)
    else:
        model = Mdl(input_shape=(224, 224, 3))

    # Define the optimizer
    Optim = config.get('optimizer')
    optimizer = Optim(**config.get('optimizer_params'))

    # Set model names
    model_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    model_name = config.get('name')
    model_filename = '{}-tf-{}'.format(model_name, model_id)

    # Define save checkpoint callback
    cp_callback = ModelHdf5Checkpoint(model_dir, model_filename)
    # Define save custom loggers callback
    lg_callback = LoggersCallback(model_dir + model_filename)

    # Define generate tensorboard log callback
    tb_callback = TensorBoard(
        log_dir='./tensorboard_logs/{}'.format(model_filename))

    # Compile the model and generate computation graph
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # Start training
    model.fit(
        train_image,
        train_label,
        epochs=config.get('total_epochs'),
        callbacks=[
            cp_callback,
            tb_callback,
            lg_callback,
        ],
        batch_size=config.get('batch_size'),
        validation_data=(val_image, val_label),
        verbose=1,
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