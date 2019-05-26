import time
from datetime import datetime
import random
import argparse
import os

import tensorflow as tf

from models import make_discriminator_model, make_generator_model
from pool import ImagePool

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LEARNING_RATE = 0.0002
BETA_1 = 0.5
LAMBDA = 10.0
LAMBDA_ID = 0.5
POOL_SIZE = 50
EPOCHS = 100
SHUFFLE_SIZE = 10000


def main():
    parser = argparse.ArgumentParser(description='Convert TFRecords for CycleGAN dataset.')
    parser.add_argument(
        '--dataset', help='The name of the dataset', required=True)
    parser.add_argument(
        '--batch_size', help='The batch size of input data', default='2')
    args = parser.parse_args()

    loss_gen_a2b_metrics = tf.keras.metrics.Mean('loss_gen_a2b_metrics', dtype=tf.float32)
    loss_gen_b2a_metrics = tf.keras.metrics.Mean('loss_gen_b2a_metrics', dtype=tf.float32)
    loss_dis_b_metrics = tf.keras.metrics.Mean('loss_dis_b_metrics', dtype=tf.float32)
    loss_dis_a_metrics = tf.keras.metrics.Mean('loss_dis_a_metrics', dtype=tf.float32)
    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    fake_pool_b2a = ImagePool(POOL_SIZE)
    fake_pool_a2b = ImagePool(POOL_SIZE)

    def calc_gan_loss(prediction, is_real):
        # Typical GAN loss to set objectives for generator and discriminator
        if is_real:
            return mse_loss(prediction, tf.ones_like(prediction))
        else:
            return mse_loss(prediction, tf.zeros_like(prediction))

    def calc_cycle_loss(reconstructed_images, real_images):
        # Cycle loss to make sure reconstructed image looks real
        return mae_loss(reconstructed_images, real_images)

    def calc_identity_loss(identity_images, real_images):
        # Identity loss to make sure generator won't do unnecessary change
        # Ideally, feeding a real image to generator should generate itself
        return mae_loss(identity_images, real_images)

    generator_a2b = make_generator_model(n_blocks=9)
    generator_b2a = make_generator_model(n_blocks=9)
    discriminator_b = make_discriminator_model()
    discriminator_a = make_discriminator_model()
    optimizer_gen_a2b = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
    optimizer_dis_b = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
    optimizer_gen_b2a = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
    optimizer_dis_a = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)

    checkpoint_dir = './checkpoints-{}'.format(args.dataset)
    checkpoint = tf.train.Checkpoint(generator_a2b=generator_a2b,
                                     generator_b2a=generator_b2a,
                                     discriminator_b=discriminator_b,
                                     discriminator_a=discriminator_a,
                                     optimizer_gen_a2b=optimizer_gen_a2b,
                                     optimizer_dis_b=optimizer_dis_b,
                                     optimizer_gen_b2a=optimizer_gen_b2a,
                                     optimizer_dis_a=optimizer_dis_a,
                                     step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    def train_step(images_a, images_b):
        real_a = images_a
        real_b = images_b

        # By default, the resources held by a GradientTape are released as soon as GradientTape.gradient()
        # method is called. To compute multiple gradients over the same computation, create a persistent gradient tape.
        # This allows multiple calls to the gradient() method as resources are released
        # when the tape object is garbage collected.
        with tf.GradientTape() as tape_gen_a2b, tf.GradientTape() as tape_gen_b2a, tf.GradientTape() as tape_dis_b, tf.GradientTape() as tape_dis_a:
            # Cycle A -> B -> A
            fake_a2b = generator_a2b(real_a, training=True)
            recon_b2a = generator_b2a(fake_a2b, training=True)
            # Cycle B -> A -> B
            fake_b2a = generator_b2a(real_b, training=True)
            recon_a2b = generator_a2b(fake_b2a, training=True)

            # Use real B to generate B should be identical
            identity_a2b = generator_a2b(real_b, training=True)
            identity_b2a = generator_b2a(real_a, training=True)
            loss_identity_a2b = calc_identity_loss(identity_a2b, real_b)
            loss_identity_b2a = calc_identity_loss(identity_b2a, real_a)

            # Generator A2B tries to trick Discriminator B that the generated image is B
            loss_gan_gen_a2b = calc_gan_loss(discriminator_b(fake_a2b), True)
            # Generator B2A tries to trick Discriminator A that the generated image is A
            loss_gan_gen_b2a = calc_gan_loss(discriminator_a(fake_b2a), True)
            loss_cycle_a2b2a = calc_cycle_loss(recon_b2a, real_a)
            loss_cycle_b2a2b = calc_cycle_loss(recon_a2b, real_b)

            # Total generator loss
            loss_total_gen_a2b = loss_gan_gen_a2b + (loss_cycle_a2b2a + loss_cycle_b2a2b) * LAMBDA + loss_identity_a2b * LAMBDA * LAMBDA_ID
            loss_total_gen_b2a = loss_gan_gen_b2a + (loss_cycle_a2b2a + loss_cycle_b2a2b) * LAMBDA + loss_identity_b2a * LAMBDA * LAMBDA_ID

            fake_b2a_from_pool = fake_pool_b2a.query(fake_b2a)
            # Discriminator A should classify real_a as A
            loss_gan_dis_a_real = calc_gan_loss(discriminator_a(real_a, training=True), True)
            # Discriminator A should classify generated fake_b2a as not A
            loss_gan_dis_a_fake = calc_gan_loss(discriminator_a(fake_b2a_from_pool, training=True), False)

            fake_a2b_from_pool = fake_pool_a2b.query(fake_a2b)
            # Discriminator B should classify real_b as B
            loss_gan_dis_b_real = calc_gan_loss(discriminator_b(real_b, training=True), True)
            loss_gan_dis_b_fake = calc_gan_loss(discriminator_b(fake_a2b_from_pool, training=True), False)

            # Total discriminator loss
            loss_dis_a = (loss_gan_dis_a_real + loss_gan_dis_a_fake) * 0.5
            loss_dis_b = (loss_gan_dis_b_real + loss_gan_dis_b_fake) * 0.5

        gradient_gen_a2b = tape_gen_a2b.gradient(loss_total_gen_a2b, generator_a2b.trainable_variables)
        gradient_gen_b2a = tape_gen_b2a.gradient(loss_total_gen_b2a, generator_b2a.trainable_variables)
        gradient_dis_a = tape_dis_a.gradient(loss_dis_a, discriminator_a.trainable_variables)
        gradient_dis_b = tape_dis_b.gradient(loss_dis_b, discriminator_b.trainable_variables)

        optimizer_gen_a2b.apply_gradients(zip(gradient_gen_a2b, generator_a2b.trainable_variables))
        optimizer_gen_b2a.apply_gradients(zip(gradient_gen_b2a, generator_b2a.trainable_variables))
        optimizer_dis_a.apply_gradients(zip(gradient_dis_a, discriminator_a.trainable_variables))
        optimizer_dis_b.apply_gradients(zip(gradient_dis_b, discriminator_b.trainable_variables))

        loss_gen_a2b_metrics(loss_total_gen_a2b)
        loss_gen_b2a_metrics(loss_total_gen_b2a)
        loss_dis_a_metrics(loss_dis_a)
        loss_dis_b_metrics(loss_dis_b)
        tf.print('loss_total_gen_a2b: ', loss_total_gen_a2b, ' loss_total_gen_b2a: ', loss_total_gen_b2a, ' loss_dis_b: ', loss_dis_b, ' loss_dis_a: ', loss_dis_a)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/horse2zebra/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Eager mode
    def train(dataset, epochs):
        for epoch in range(1, epochs+1):
            start = time.time()

            for batch in dataset:
                train_step(batch[0], batch[1])

            with train_summary_writer.as_default():
                tf.summary.scalar('loss_total_gen_a2b', loss_gen_a2b_metrics.result(), step=epoch)
                tf.summary.scalar('loss_total_gen_b2a', loss_gen_b2a_metrics.result(), step=epoch)
                tf.summary.scalar('loss_dis_b', loss_dis_b_metrics.result(), step=epoch)
                tf.summary.scalar('loss_dis_a', loss_dis_a_metrics.result(), step=epoch)

            tf.print('Epoch ', epoch,
                     ' avg loss_total_gen_a2b: ', loss_gen_a2b_metrics.result(),
                     ' avg loss_total_gen_b2a: ', loss_gen_b2a_metrics.result(),
                     ' avg loss_dis_b: ', loss_dis_b_metrics.result(),
                     ' avg loss_dis_a: ', loss_dis_a_metrics.result())
            loss_gen_a2b_metrics.reset_states()
            loss_gen_b2a_metrics.reset_states()
            loss_dis_b_metrics.reset_states()
            loss_dis_a_metrics.reset_states()

            checkpoint.step.assign_add(1)
            if epoch % 5 == 0:
                save_path = checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

    def make_dataset(filepath):
        raw_dataset = tf.data.TFRecordDataset(filepath)

        image_feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
        }

        def preprocess_image(encoded_image):
            image = tf.image.decode_jpeg(encoded_image, 3)
            # resize to 256x256
            image = tf.image.resize(image, [256, 256])
            # normalize from 0-255 to -1 ~ +1
            image = image / 127.5 - 1
            return image

        def parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            features = tf.io.parse_single_example(example_proto, image_feature_description)
            encoded_image = features['image/encoded']
            image = preprocess_image(encoded_image)
            return image

        parsed_image_dataset = raw_dataset.map(parse_image_function)
        return parsed_image_dataset

    train_a = make_dataset('tfrecords/{}/trainA.tfrecord'.format(args.dataset))
    train_b = make_dataset('tfrecords/{}/trainB.tfrecord'.format(args.dataset))
    combined_dataset = tf.data.Dataset.zip((train_a, train_b)).shuffle(SHUFFLE_SIZE).batch(int(args.batch_size))

    # for local testing
    # seed1 = tf.random.normal([2, 256, 256, 3])
    # seed2 = tf.random.normal([2, 256, 256, 3])
    # combined_dataset = [(seed1, seed2)]
    # EPOCHS = 2

    train(combined_dataset, EPOCHS)
    print('Finished training.')


if __name__ == '__main__':
    main()
