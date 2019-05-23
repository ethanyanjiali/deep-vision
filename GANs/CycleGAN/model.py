"""
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
https://arxiv.org/pdf/1703.10593.pdf
"""
import tensorflow as tf


def make_bn_and_leaky_relu():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    return model


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,))
        self.bn_relu_dense = make_bn_and_leaky_relu()
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))
        self.conv1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn_relu_conv1 = make_bn_and_leaky_relu()
        self.conv2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn_relu_conv2 = make_bn_and_leaky_relu()
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

    def call(self, inputs, training=None, mask=None):
        x = self.dense(inputs)
        x = self.bn_relu_dense(x)
        x = self.reshape(x)

        x = self.conv1(x)
        x = self.bn_relu_conv1(x)

        x = self.conv2(x)
        x = self.bn_relu_conv2(x)

        x = self.conv3(x)
        return x


def make_leaky_relu_and_dropout():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    return model


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])
        self.relu_dropout1 = make_leaky_relu_and_dropout()

        self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])
        self.relu_dropout2 = make_leaky_relu_and_dropout()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.relu_dropout1(x)

        x = self.conv2(x)
        x = self.relu_dropout2(x)

        x = self.flatten(x)
        x = self.dense(x)
        return x


def make_gan_loss():
    return


def make_cycle_loss():
    return


def make_identity_loss():
    return


class CycleGAN(tf.keras.Model):
    def __init__(self, learning_rate, beta_1):
        super(CycleGAN, self).__init__()
        self.netG_A = Generator()
        self.netG_B = Generator()
        self.netD_A = Discriminator()
        self.netD_B = Discriminator()
        self.loss_gan = make_gan_loss()
        self.loss_cycle = make_cycle_loss()
        self.loss_identity = make_identity_loss()
        self.optimizer_G = tf.keras.optimizers.Adam(learning_rate, beta_1)
        self.optimizer_D = tf.keras.optimizers.Adam(learning_rate, beta_1)