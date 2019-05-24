import time
import random
import argparse
import os

import tensorflow as tf

from models import make_discriminator_model, make_generator_model

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LEARNING_RATE = 0.0002
BETA_1 = 0.5
LAMBDA_A = 10.0
LAMBDA_B = 10.0
LAMBDA_ID = 0.5
POOL_SIZE = 50
EPOCHS = 100
BATCH_SIZE = 32
SHUFFLE_SIZE = 10000


def main():
    parser = argparse.ArgumentParser(description='Convert iFood 2018 dataset to TFRecord files.')
    parser.add_argument(
        '--dataset', help='The name of the dataset')
    args = parser.parse_args()

    mse_loss = tf.keras.losses.MeanSquaredError()
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    image_pool_fake_A = []
    image_pool_fake_B = []

    def query_image_pool(fake_images, fake_pool):
        if len(fake_pool) < POOL_SIZE:
            fake_pool.append(fake_images)
            return fake_images
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, POOL_SIZE-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake_images
                return temp
            else:
                return fake_images

    def query_image_pool_A(fake_images):
        return query_image_pool(fake_images, image_pool_fake_A)

    def query_image_pool_B(fake_images):
        return query_image_pool(fake_images, image_pool_fake_B)

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

    netG_A = make_generator_model(n_blocks=9)
    netG_B = make_generator_model(n_blocks=9)
    netD_A = make_discriminator_model()
    netD_B = make_discriminator_model()
    optimizer_G_A = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
    optimizer_D_A = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
    optimizer_G_B = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
    optimizer_D_B = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)

    checkpoint_dir = './checkpoints-{}'.format(args.dataset)
    checkpoint = tf.train.Checkpoint(netG_A=netG_A,
                                     netG_B=netG_B,
                                     netD_A=netD_A,
                                     netD_B=netD_B,
                                     optimizer_G_A=optimizer_G_A,
                                     optimizer_D_A=optimizer_D_A,
                                     optimizer_G_B=optimizer_G_B,
                                     optimizer_D_B=optimizer_D_B,
                                     step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    @tf.function
    def train_step(images_A, images_B):
        real_A = images_A
        real_B = images_B

        with tf.GradientTape() as tapeG_A, tf.GradientTape() as tapeG_B:
            with tf.GradientTape() as tapeD_A, tf.GradientTape() as tapeD_B:
                # Cycle A -> B -> A
                fake_B = netG_A(real_A)
                recon_A = netG_B(fake_B)
                # Cycle B -> A -> B
                fake_A = netG_B(real_B)
                recon_B = netG_A(fake_A)

                # Use real B to generate B should be identical
                identity_A = netG_A(real_B)
                identity_B = netG_B(real_A)
                loss_identity_A = calc_identity_loss(identity_A, real_B)
                loss_identity_B = calc_identity_loss(identity_B, real_A)

                # Generator tries to trick Discriminator
                loss_gan_G_A = calc_gan_loss(netD_A(fake_B), True)
                loss_gan_G_B = calc_gan_loss(netD_B(fake_A), True)
                loss_cycle_A = calc_cycle_loss(recon_A, real_A)
                loss_cycle_B = calc_cycle_loss(recon_B, real_B)

                loss_G_A = loss_gan_G_A + loss_cycle_A * LAMBDA_A + loss_identity_A * LAMBDA_A * LAMBDA_ID
                loss_G_B = loss_gan_G_B + loss_cycle_B * LAMBDA_B + loss_identity_B * LAMBDA_B * LAMBDA_ID

                fake_A_to_inspect = query_image_pool_A(fake_A)
                decision_B_real = netD_B(real_A)
                decision_B_fake = netD_B(fake_A_to_inspect)
                # For discriminator, true is true, false is false
                loss_gan_D_B_real = calc_gan_loss(decision_B_real, True)
                loss_gan_D_B_fake = calc_gan_loss(decision_B_fake, False)
                loss_D_B = (loss_gan_D_B_real + loss_gan_D_B_fake) * 0.5

                fake_B_to_inspect = query_image_pool_B(fake_B)
                decision_A_real = netD_A(real_B)
                decision_A_fake = netD_A(fake_B_to_inspect)
                # For discriminator, true is true, false is false
                loss_gan_D_A_real = calc_gan_loss(decision_A_real, True)
                loss_gan_D_A_fake = calc_gan_loss(decision_A_fake, False)
                loss_D_A = (loss_gan_D_A_real + loss_gan_D_A_fake) * 0.5

        gradientG_A = tapeG_A.gradient(loss_G_A, netG_A.trainable_variables)
        gradientG_B = tapeG_B.gradient(loss_G_B, netG_B.trainable_variables)
        gradientD_A = tapeD_A.gradient(loss_D_A, netD_A.trainable_variables)
        gradientD_B = tapeD_B.gradient(loss_D_B, netD_B.trainable_variables)

        optimizer_G_A.apply_gradients(zip(gradientG_A, netG_A.trainable_variables))
        optimizer_D_A.apply_gradients(zip(gradientD_A, netD_A.trainable_variables))
        optimizer_G_B.apply_gradients(zip(gradientG_B, netG_A.trainable_variables))
        optimizer_D_B.apply_gradients(zip(gradientD_B, netD_A.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(1, epochs+1):
            start = time.time()

            for batch in dataset:
                print(batch[0].shape, batch[1].shape)
                train_step(batch[0], batch[1])

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
            # normalize from 0-255 to 0-1
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

    train_A = make_dataset('tfrecords/{}/trainA.tfrecord'.format(args.dataset))
    train_B = make_dataset('tfrecords/{}/trainB.tfrecord'.format(args.dataset))
    combined_dataset = tf.data.Dataset.zip((train_A, train_B)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)

    train(combined_dataset, EPOCHS)
    print('Finished training.')


if __name__ == '__main__':
    main()
