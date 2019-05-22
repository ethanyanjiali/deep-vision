import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras import regularizers

# [1] Deep Residual Learning for Image Recognition
# https://arxiv.org/pdf/1512.03385.pdf

# We use a weight decay of 0.0001..."" resnet34.[1]
weight_decay = 0.0001


def ResNet152(input_shape):
    inputs = layers.Input(shape=input_shape)
    # floor[(224 - 7 + 2 * 3) / 2] + 1 = 112, it becomes 112x112x64
    # here we maunally add padding=3
    x = layers.ZeroPadding2D(padding=(3, 3))(inputs)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    # "We adopt batch normalization (BN) right after each convolution and before activation"[1]
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    # "3×3 max pool, stride 2" in Table 1 of [1]
    # (112 - 3) / 2 + 1 = 56
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # this refers to Table 1 in [1] 152-layer column
    x = _residual_blocks(x, 3, 64, 256, strides=1)
    x = _residual_blocks(x, 8, 128, 512, strides=2)
    x = _residual_blocks(x, 36, 256, 1024, strides=2)
    x = _residual_blocks(x, 3, 512, 2048, strides=2)

    # average poorling with a 1000-way softmax
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # this avg pool has already flatten the output
    outputs = layers.Dense(1000, activation='softmax')(x)

    model = Model(inputs, outputs, name='resnet152')
    return model


def _residual_blocks(inputs, num_blocks, out_channels_1, out_channels_2,
                     strides):
    x = BottleneckBlock(
        inputs,
        out_channels_1,
        out_channels_2,
        strides,
        downsample=True,
    )
    for i in range(1, num_blocks):
        x = BottleneckBlock(
            x,
            out_channels_1,
            out_channels_2,
        )
    return x


def BottleneckBlock(inputs,
                    out_channels_1,
                    out_channels_2,
                    strides=1,
                    downsample=False):
    identity = inputs
    # for the first block in a group, if it needs downsample, we need to do a projection
    # to shrink the dimension
    if downsample:
        identity = layers.Conv2D(
            out_channels_2,
            1,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(weight_decay),
        )(inputs)
        identity = layers.BatchNormalization(axis=3)(identity)

    # According to Figure 5 Right in [1]
    # 1x1 reduce dimension
    x = layers.Conv2D(
        out_channels_1,
        1,
        strides=strides,
        padding='same',
        # with batch norm below, there's no need to use bias here
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
    )(inputs)
    # tensor flow is channel-last, so bach norm axis = 3
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    # 3x3
    x = layers.Conv2D(
        out_channels_1,
        3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    # 1x1 increase dimension
    x = layers.Conv2D(
        out_channels_2,
        1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
    )(x)

    # add identity to output
    x = layers.add([x, identity])
    x = layers.Activation('relu')(x)

    return x
