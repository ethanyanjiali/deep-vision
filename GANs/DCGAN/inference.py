import tensorflow as tf
import matplotlib.pyplot as plt

from models import Generator, Discriminator


def main():
    generator = Generator()
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()