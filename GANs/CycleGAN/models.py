"""
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
https://arxiv.org/pdf/1703.10593.pdf
"""
import tensorflow as tf


class ReflectionPad2d(tf.keras.layers.Layer):
    def __init__(self, padding, **kwargs):
        super(ReflectionPad2d, self).__init__(**kwargs)
        self.padding = [[0, 0], [padding, padding], [padding, padding], [0, 0]]

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.padding, 'REFLECT')


class ResNetBlock(tf.keras.Model):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.padding1 = ReflectionPad2d(1)
        self.conv1 = tf.keras.layers.Conv2D(dim, (3, 3), padding='valid', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.padding2 = ReflectionPad2d(1)
        self.conv2 = tf.keras.layers.Conv2D(dim, (3, 3), padding='valid', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.padding1(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.padding2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        outputs = inputs + x
        return outputs


def make_generator_model(n_blocks):
    # 6 residual blocks
    # c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
    # 9 residual blocks
    # c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
    model = tf.keras.Sequential()

    # Encoding
    model.add(ReflectionPad2d(3, input_shape=(256, 256, 3)))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), strides=(1, 1), padding='valid', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Transformation
    for i in range(n_blocks):
        model.add(ResNetBlock(256))

    # Decoding
    model.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(ReflectionPad2d(3))
    model.add(tf.keras.layers.Conv2D(3, (7, 7), strides=(1, 1), padding='valid', activation='tanh'))

    return model


def make_discriminator_model():
    # C64-C128-C256-C512
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'), input_shape=(256, 256, 3))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(512, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # This last conv net is the PatchGAN
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39#issuecomment-305575964
    # https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m
    model.add(tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same'))

    return model
