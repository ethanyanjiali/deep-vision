import glob
import os

import tensorflow as tf
import matplotlib.pyplot as plt

from models import make_generator_model, make_discriminator_model


def main():
    generator_a2b = make_generator_model(n_blocks=9)
    generator_b2a = make_generator_model(n_blocks=9)

    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(generator_a2b=generator_a2b,
                                     generator_b2a=generator_b2a)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    files = glob.glob(os.path.join('datasets', 'horse2zebra', 'trainA', '*'))
    with open(files[0], 'rb') as f:
        content = f.read()
    original = tf.image.decode_jpeg(content)
    float_original = tf.cast(original, tf.float32)
    inputs = float_original / 127.5 - 1
    inputs = tf.expand_dims(inputs, 0)
    # outputs = generator_b2a(inputs)
    outputs = generator_a2b(inputs)
    generated = outputs[0]
    generated = (generated + 1) * 127.5
    generated = tf.cast(generated, tf.uint8)
    # print(generated)
    # plt.imshow(original)
    # plt.show()
    # plt.imshow(generated)
    # plt.show()
    # print(netD_A(inputs))
    # print(netD_A(outputs).shape)
    print(len(generator_a2b.trainable_variables))
    print(len(generator_b2a.trainable_variables))


if __name__ == '__main__':
    main()

