import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

    def call(self, inputs, training=None, mask=None):
        return inputs


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

    def call(self, inputs, training=None, mask=None):
        return inputs


class CycleGAN(tf.keras.Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.netG_A = Generator()
        self.netG_B = Generator()
        self.netD_A = Discriminator()
        self.netD_B = Discriminator()