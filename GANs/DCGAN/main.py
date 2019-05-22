"""
UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
https://arxiv.org/pdf/1511.06434.pdf
"""
import os

import tensorflow as tf
import glob
import time
import matplotlib.pyplot as plt
from models import Generator, Discriminator

print(tf.__version__)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16


def main():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Try to recognize real ones as real, and fake ones as fake
    def calc_discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    # Try to make fake ones look real
    def calc_generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_image = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_image, training=True)

            generator_loss = calc_generator_loss(fake_output)
            discriminator_loss = calc_discriminator_loss(real_output, fake_output)

        generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    def generate_and_save_images(model, epoch, test_input):
        predictions = model(test_input, training=False)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

    def train(dataset, epochs):
        for epoch in range(1, epochs+1):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            generate_and_save_images(generator, epoch, seed)

            if epoch % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

    train(train_dataset, EPOCHS)


if __name__ == '__main__':
    main()