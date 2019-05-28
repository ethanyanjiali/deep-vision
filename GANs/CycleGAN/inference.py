import glob
import os

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from models import make_generator_model, make_discriminator_model


def main():
    generator_a2b = make_generator_model(n_blocks=9)
    generator_b2a = make_generator_model(n_blocks=9)

    checkpoint_dir = './checkpoints-horse2zebra'
    checkpoint = tf.train.Checkpoint(generator_a2b=generator_a2b,
                                     generator_b2a=generator_b2a)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    files = glob.glob(os.path.join('datasets', 'horse2zebra', 'testB', '*'))
    for idx in range(120):
        with open(files[idx], 'rb') as f:
            content = f.read()
        original = tf.image.decode_jpeg(content)
        float_original = tf.cast(original, tf.float32)
        inputs = float_original / 127.5 - 1
        inputs = tf.expand_dims(inputs, 0)
        outputs = generator_b2a(inputs)
        # outputs = generator_a2b(inputs)
        generated = outputs[0]
        generated = (generated + 1) * 127.5
        generated = tf.cast(generated, tf.uint8)
        # plt.imshow(original)
        # plt.show()
        # plt.imshow(generated)
        # plt.show()
        # tf.saved_model.save(generator_a2b, "./saved_models/horse2zebra/1/")
        im1 = Image.fromarray(original.numpy())
        im1.save('samples/b2a_{}_original.JPEG'.format(idx))
        im2 = Image.fromarray(generated.numpy())
        im2.save('samples/b2a_{}_generated.JPEG'.format(idx))


if __name__ == '__main__':
    main()

