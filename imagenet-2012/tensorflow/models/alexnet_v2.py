import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, ZeroPadding2D, Layer
from tensorflow.keras import Model, Input

# [1] One weird trick for parallelizing convolutional neural networks
# https://arxiv.org/pdf/1404.5997.pdf


class LocalResponseNorm(Layer):
    def __init__(self, output_dim, **kwargs):
        super(LocalResponseNorm, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True,
        )
        # Be sure to call this at the end
        super(LocalResponseNorm, self).build(input_shape)

    def call(self, x):
        return tf.nn.local_response_normalization(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def AlexNetV2(input_shape):
    # formula
    # [conv layer]
    # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
    # padding = ((output_size - 1) * stride) + kernel_size - input_size) / 2
    # [pooling layer]
    # output_size = (input_size - kernel_size) / stride + 1
    # where input_size and output_size are the square image side length

    inputs = Input(input_shape)
    # To use 'valid' padding and output 55x55 size, we need to pad the input first
    # Alex got it wrong in his origina paper, the input should be 227x227 instead of 224x224
    x = ZeroPadding2D(3)(inputs)
    # valid padding, so no padding
    # output = (227 - 11 + 2 * 0) / 4 + 1 = 55
    x = Conv2D(64, 11, strides=4, padding='valid', activation='relu')(x)
    x = LocalResponseNorm(64)(x)
    # output = (55 - 3) / 2 + 1 = 27
    x = MaxPool2D(3, 2)(x)
    # same padding, so padding = kernel / 2 = 2
    # output = (27 - 5 + 2 * 2) / 1 + 1 = 27
    x = Conv2D(192, 5, strides=1, padding='same', activation='relu')(x)
    x = LocalResponseNorm(192)(x)
    # output = (27 - 3) / 2 + 1 = 13
    x = MaxPool2D(3, 2)(x)
    # output = (13 - 3 + 2 * 1) / 1 + 1 = 13
    x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)
    # output = (13 - 3 + 2 * 1) / 1 + 1 = 13
    x = Conv2D(384, 3, strides=1, padding='same', activation='relu')(x)
    # output = (13 - 3 + 2 * 1) / 1 + 1 = 13
    x = Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    # output = (13 - 3) / 2 + 1 = 6
    x = MaxPool2D(3, 2)(x)

    # flatten from 6x6x256 to 4096
    x = Flatten()(x)

    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    # finally a 1000-ways softmax layer as classifier
    outputs = Dense(1000, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
