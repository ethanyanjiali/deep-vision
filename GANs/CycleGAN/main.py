import tensorflow as tf
import matplotlib.pyplot as plt
from models import make_discriminator_model, make_generator_model

print(tf.__version__)

LEARNING_RATE = 0.0002
BETA_1 = 0.5


def main():
    def make_gan_loss():
        return

    def make_cycle_loss():
        return

    def make_identity_loss():
        return

    netG_A = make_generator_model(n_blocks=6)
    netG_B = make_generator_model(n_blocks=6)
    netD_A = make_discriminator_model()
    netD_B = make_discriminator_model()
    loss_gan = make_gan_loss()
    loss_cycle = make_cycle_loss()
    loss_identity = make_identity_loss()
    optimizer_G = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
    optimizer_D = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)

    seed = tf.random.normal([1, 128, 128, 3])
    generated_image = netG_A(seed)
    # print(generated_image[0].shape)

    plt.imshow(generated_image[0])
    plt.show()


if __name__ == '__main__':
    main()