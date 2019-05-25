import glob
import os

import tensorflow as tf
import matplotlib.pyplot as plt

from models import make_generator_model


def main():
    netG_A = make_generator_model(n_blocks=9)

    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(netG_A=netG_A)
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
    original = tf.cast(original, tf.float32)
    inputs = original / 127.5 - 1
    inputs = tf.expand_dims(inputs, 0)
    outputs = netG_A(inputs)
    generated = outputs[0]
    generated = (generated + 1) * 127.5
    generated = tf.cast(generated, tf.uint8)
    print(generated)
    plt.imshow(generated)
    plt.show()


if __name__ == '__main__':
    main()

