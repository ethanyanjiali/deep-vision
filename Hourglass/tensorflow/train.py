import os
from datetime import datetime

import tensorflow as tf

from hourglass import StackedHourglassNetwork
from preprocess import Preprocessor

IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 32
TOTAL_EPOCHS = 100
HEATMAP_SHAPE = (64, 64, 16)
TF_RECORDS_DIR = './dataset/tfrecords_mpii/'


class Trainer(object):
    def __init__(self,
                 model,
                 epochs,
                 global_batch_size,
                 strategy,
                 initial_learning_rate=0.001):
        self.model = model
        self.epochs = epochs
        self.strategy = strategy
        self.initial_learning_rate = initial_learning_rate
        self.global_batch_size = global_batch_size
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate)
        self.model = model

    def lr_decay(self, epoch):
        if epoch < 50:
            return self.initial_learning_rate
        if epoch >= 50 and epoch < 75:
            return self.initial_learning_rate / 10.0
        if epoch >= 75:
            return self.initial_learning_rate / 100.0

    def compute_loss(self, labels, logits):
        loss = tf.reduce_sum(self.loss_object(
            labels, logits)) * (1. / self.global_batch_size)
        return loss

    def train_step(self, inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss = self.compute_loss(labels, logits)

        grads = tape.gradient(
            target=loss, sources=self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        return loss

    def val_step(self, inputs):
        images, labels = inputs
        logits = self.model(images, training=False)
        loss = self.compute_loss(labels, logits)
        return loss

    def run(self, train_dist_dataset, val_dist_dataset):
        @tf.function
        def distributed_train_epoch(dataset):
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in dataset:
                per_replica_loss = self.strategy.experimental_run_v2(
                    self.train_step, args=(one_batch, ))
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_loss += batch_loss
                num_train_batches += 1
                tf.print('Trained batch', num_train_batches, 'batch loss',
                         batch_loss, 'epoch total loss', total_loss)
            return total_loss, num_train_batches

        @tf.function
        def distributed_val_epoch(dataset):
            total_loss = 0.0
            num_val_batches = 0.0
            for one_batch in dataset:
                per_replica_loss = self.strategy.experimental_run_v2(
                    self.val_step, args=(one_batch, ))
                num_val_batches += 1
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_loss += batch_loss
            return total_loss, num_val_batches

        for epoch in range(1, self.epochs + 1):
            self.optimizer.learning_rate = self.lr_decay(epoch)

            train_total_loss, num_train_batches = distributed_train_epoch(
                train_dist_dataset)
            print('Epoch {} train loss {}'.format(
                epoch, train_total_loss / num_train_batches))

            val_total_loss, num_val_batches = distributed_val_epoch(
                val_dist_dataset)
            print('Epoch {} val loss {}'.format(
                epoch, val_total_loss / num_val_batches))

            if epoch % 5 == 0:
                self.model.save('./models/model-v1.0.0-epoch-%d.h5' % epoch)
                print("Model saved")

        self.model.save(
            './models/model-final-v1.0.0-epoch-%d.h5' % self.epochs)
        print("Model saved")


def create_dataset(tfrecords, batch_size, is_train):
    preprocess = Preprocessor(is_train, IMAGE_SHAPE, HEATMAP_SHAPE)

    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(preprocess, num_parallel_calls=4)

    if is_train:
        dataset = dataset.shuffle(1024)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=512)

    return dataset


def main():
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * BATCH_SIZE
    # train_dataset = create_dataset(
    #     os.path.join(TF_RECORDS_DIR, 'train*'),
    #     global_batch_size,
    #     is_train=True)
    # val_dataset = create_dataset(
    #     os.path.join(TF_RECORDS_DIR, 'val*'),
    #     global_batch_size,
    #     is_train=False)

    if not os.path.exists(os.path.join('./models')):
        os.makedirs(os.path.join('./models/'))

    with strategy.scope():
        # train_dist_dataset = strategy.experimental_distribute_dataset(
        #     train_dataset)
        # val_dist_dataset = strategy.experimental_distribute_dataset(
        #     val_dataset)

        model = StackedHourglassNetwork(IMAGE_SHAPE, 4, 1, HEATMAP_SHAPE[2])
        model.summary()

        trainer = Trainer(model, TOTAL_EPOCHS, global_batch_size, strategy)

        print('Start training...')
        # trainer.run(train_dist_dataset, val_dist_dataset)


if __name__ == "__main__":
    main()