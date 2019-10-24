import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, DepthwiseConv2D, AveragePooling2D, Dense


# The SeparableConv2D implementation in TF2 doesn't have BN
class SeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SeparableConv2D, self).__init__()
        self.dwconv = DepthwiseConv2D(
            kernel_size=kernel_size, strides=strides, padding='same')
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        self.pwconv = Conv2D(
            filters=filters, kernel_size=1, strides=1, padding='same')

    def call(self, input, training=None):
        x = self.dwconv(input)
        x = self.bn1(x, training)
        x = self.relu(x)
        x = self.pwconv(x)
        x = self.bn2(x, training)
        x = self.relu(x)
        return x


def MobileNetV1(input_shape, alpha=1):
    return Sequential([
        # As shown in Table 1
        # Filter shpe 3x3x3x32, Inputs Size 224x224x3, strides=2
        Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            input_shape=input_shape),
        # Filter shpe 3x3x32 dw, Inputs Size 112x112x32
        # Filter shpe 1x1x32x64 pw, Inputs Size 112x112x64
        SeparableConv2D(filters=64, kernel_size=3, strides=1),
        # Filter shpe 3x3x64 dw, Inputs Size 112x112x64, strides=2
        # Filter shpe 1x1x64x128 pw, Inputs Size 56x56x64
        SeparableConv2D(filters=128, kernel_size=3, strides=2),
        # Filter shpe 3x3x128 dw, Inputs Size 56x56x128
        # Filter shpe 1x1x128x128 pw, Inputs Size 56x56x128
        SeparableConv2D(filters=128, kernel_size=3, strides=1),
        # Filter shpe 3x3x128 dw, Inputs Size 56x56x128
        # Filter shpe 1x1x128x256 pw, Inputs Size 28x28x128, strides=2
        SeparableConv2D(filters=256, kernel_size=3, strides=2),
        # Filter shpe 3x3x256 dw, Inputs Size 28x28x256
        # Filter shpe 1x1x256x256 pw, Inputs Size 28x28x256
        SeparableConv2D(filters=256, kernel_size=3, strides=1),
        # Filter shpe 3x3x256 dw, Inputs Size 28x28x256, strides=2
        # Filter shpe 1x1x256x512 pw, Inputs Size 14x14x512
        SeparableConv2D(filters=512, kernel_size=3, strides=2),
        # 5x same layers with 512 filters
        # Filter shpe 3x3x512 dw, Inputs Size 14x14x512
        # Filter shpe 1x1x512x512 pw, Inputs Size 14x14x512
        SeparableConv2D(filters=512, kernel_size=3, strides=1),
        SeparableConv2D(filters=512, kernel_size=3, strides=1),
        SeparableConv2D(filters=512, kernel_size=3, strides=1),
        SeparableConv2D(filters=512, kernel_size=3, strides=1),
        SeparableConv2D(filters=512, kernel_size=3, strides=1),
        # Filter shpe 3x3x512 dw, Inputs Size 14x14x512, strides=2
        # Filter shpe 1x1x512x1024 pw, Inputs Size 7x7x512
        SeparableConv2D(filters=1024, kernel_size=3, strides=2),
        # Filter shpe 3x3x1024 dw, Inputs Size 7x7x1024
        # Filter shpe 1x1x1024x1024 pw, Inputs Size 7x7x1024
        SeparableConv2D(filters=1024, kernel_size=3, strides=1),
        # Pool 7X7, Input Size 7x7x1024
        AveragePooling2D(pool_size=7, strides=1),
        # FC 1024x1000
        Dense(units=1000),
    ])
