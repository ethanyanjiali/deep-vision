import argparse
import time
import pickle

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import mnist
import numpy as np
from models.alexnet_v2 import AlexNetV2

model_dir = './saved_models/'

training_config = {
    'alexnet2': {
        'name': 'alexnet2',
        'model': AlexNetV2,
        'batch_size': 128,
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


def run_epochs(config, checkpoint_path):
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
    # model.fit(
    #     x_train,
    #     y_train,
    #     epochs=config.get('total_epochs'),
    #     callbacks=[
    #         cp_callback,
    #         tb_callback,
    #         lg_callback,
    #     ],
    #     batch_size=config.get('batch_size'),
    #     validation_data=(x_test, y_test),
    #     verbose=1,
    # )


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