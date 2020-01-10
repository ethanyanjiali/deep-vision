import argparse
import math
import datetime
import os
import time

import tensorflow as tf
import numpy as np

from model import ObjectsAsPoints
from preprocess import Preprocessor

BATCH_SIZE = 16
TOTAL_CLASSES = 80
TOTAL_EPOCHS = 300
OUTPUT_SHAPE = (416, 416)
TF_RECORDS = './dataset/tfrecords'

tf.random.set_seed(1)


class Trainer(object):
    def __init__(self,
                 model,
                 initial_epoch,
                 epochs,
                 global_batch_size,
                 strategy,
                 initial_learning_rate=0.01):
        self.model = model
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.loss_objects = []
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate)

        # for learning rate schedule
        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.patience_count = 0
        self.max_patience = 10

    def lr_decay(self):
        """
        This effectively simulate ReduceOnPlateau learning rate schedule. Learning rate
        will be reduced by a factor of 10 if there's no improvement over [max_patience] epochs
        """
        if self.patience_count > self.max_patience:
            self.current_learning_rate /= 10.0
            self.patience_count = 0
        elif self.last_val_loss == self.lowest_val_loss:
            self.patience_count = 0
        self.patience_count += 1

        self.optimizer.learning_rate = self.current_learning_rate

    def train_step(self, inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            total_losses = []
            # iterate over all three scales
            for loss_object, y_pred, y_true in zip(self.loss_objects, outputs,
                                                   labels):
                total_loss = loss_object(y_true, y_pred)

            total_loss = tf.reduce_sum(total_losses)

        grads = tape.gradient(
            target=total_loss, sources=self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return total_loss

    def val_step(self, inputs):
        images, labels = inputs

        outputs = self.model(images, training=False)
        losses = []
        # iterate over all three scales
        for loss_object, y_pred, y_true in zip(self.loss_objects, outputs,
                                               labels):
            loss = loss_object(y_true, y_pred)
            losses.append(loss * (1. / self.global_batch_size))
        total_loss = tf.reduce_sum(losses)

        return total_loss

    def get_current_time(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def run(self, train_dist_dataset, val_dist_dataset):
        total_steps = tf.constant(0, dtype=tf.int64)

        @tf.function
        def distributed_train_epoch(dataset, summary_writer, total_steps):
            total_loss = 0.0
            num_train_batches = tf.constant(0, dtype=tf.int64)
            for one_batch in dataset:
                per_replica_losses = self.strategy.experimental_run_v2(
                    self.train_step, args=(one_batch, ))
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                total_loss += batch_loss
                num_train_batches += 1
                tf.print('Trained batch:', num_train_batches, 'batch loss:',
                         batch_loss)
                with summary_writer.as_default():
                    tf.summary.scalar(
                        'batch train loss',
                        batch_loss,
                        step=total_steps + num_train_batches)
            return total_loss, num_train_batches

        @tf.function
        def distributed_val_epoch(dataset):
            total_loss = 0.0
            num_val_batches = tf.constant(0, dtype=tf.int64)
            for one_batch in dataset:
                per_replica_losses = self.strategy.experimental_run_v2(
                    self.val_step, args=(one_batch, ))
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                total_loss += batch_loss
                num_val_batches += 1
            return total_loss, num_val_batches

        current_time = self.get_current_time()
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        tf.print('{} Start training...'.format(current_time))
        for epoch in range(self.initial_epoch, self.epochs + 1):
            t0 = time.time()
            self.lr_decay()

            tf.print(
                '{} Started epoch {} with learning rate {}. Current LR patience count is {} epochs. Last lowest val loss is {}.'
                .format(self.get_current_time(), epoch,
                        self.current_learning_rate, self.patience_count,
                        self.lowest_val_loss))

            train_total_loss, num_train_batches = distributed_train_epoch(
                train_dist_dataset, summary_writer, total_steps)
            t1 = time.time()
            train_loss = train_total_loss / tf.cast(
                num_train_batches, dtype=tf.float32)
            tf.print(
                '{} Epoch {} train loss {}, total train batches {}, {} examples per second'
                .format(
                    self.get_current_time(), epoch, train_loss,
                    num_train_batches,
                    tf.cast(num_train_batches, dtype=tf.float32) *
                    self.global_batch_size / (t1 - t0)))
            with summary_writer.as_default():
                tf.summary.scalar('epoch train loss', train_loss, step=epoch)
            total_steps += num_train_batches

            val_total_loss, num_val_batches = distributed_val_epoch(
                val_dist_dataset)

            t2 = time.time()
            val_loss = val_total_loss / tf.cast(
                num_val_batches, dtype=tf.float32)
            tf.print(
                '{} Epoch {} val loss {}, total val batches {}, {} examples per second'
                .format(
                    self.get_current_time(), epoch, val_loss, num_val_batches,
                    tf.cast(num_val_batches, dtype=tf.float32) *
                    self.global_batch_size / (t2 - t1)))
            with summary_writer.as_default():
                tf.summary.scalar('epoch val loss', val_loss, step=epoch)

            # save model when reach a new lowest validation loss
            if val_loss < self.lowest_val_loss:
                self.save_model(epoch, val_loss)
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss

        self.save_model(self.epochs, self.last_val_loss)
        print('{} Finished.'.format(self.get_current_time()))

    def save_model(self, epoch, loss):
        model_name = './models/model-v1.0.0-epoch-{}-loss-{:.4f}.h5'.format(
            epoch, loss)
        self.model.save_weights(model_name)
        print("Model {} saved.".format(model_name))


def create_dataset(tfrecords, batch_size, is_train):
    preprocess = Preprocessor(is_train, TOTAL_CLASSES, OUTPUT_SHAPE)

    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_train:
        dataset = dataset.shuffle(512)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='checkpoint file path')
    args = parser.parse_args()

    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * BATCH_SIZE
    train_dataset = create_dataset(
        '{}/train*'.format(TF_RECORDS), global_batch_size, is_train=True)
    val_dataset = create_dataset(
        '{}/val*'.format(TF_RECORDS), global_batch_size, is_train=False)
    if not os.path.exists(os.path.join('./models')):
        os.makedirs(os.path.join('./models/'))

    with strategy.scope():
        train_dist_dataset = strategy.experimental_distribute_dataset(
            train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(
            val_dataset)
        model = ObjectsAsPoints(num_classes=TOTAL_CLASSES)
        model.summary()

        initial_epoch = 1
        if args.checkpoint:
            model.load_weights(args.checkpoint)
            initial_epoch = int(args.checkpoint.split('-')[-3]) + 1
            print('Resume training from checkpoint {} and epoch {}'.format(
                args.checkpoint, initial_epoch))

        trainer = Trainer(
            model=model,
            initial_epoch=initial_epoch,
            epochs=TOTAL_EPOCHS,
            global_batch_size=global_batch_size,
            strategy=strategy,
        )
#         trainer.run(train_dist_dataset, val_dist_dataset)


if __name__ == '__main__':
    main()
