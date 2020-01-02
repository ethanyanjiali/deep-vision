import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Softmax

import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras import regularizers

# [1] Deep Residual Learning for Image Recognition
# https://arxiv.org/pdf/1512.03385.pdf

# [2] Identity Mappings in Deep Residual Networks
# https://arxiv.org/pdf/1603.05027.pdf

# We use a weight decay of 0.0001..."" resnet34.[1]
weight_decay = 0.0001


def FeatureExtractor(input_shape):
    inputs = layers.Input(shape=input_shape, name='input_1')
    # floor[(224 - 7 + 2 * 3) / 2] + 1 = 112, it becomes 112x112x64
    # here we maunally add padding=3
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        use_bias=True,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name='conv1_conv',
    )(x)
    # In ResNetV2, activation is happened after BN, so we pool directly here
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(x)

    # this refers to Table 1 in [1] 152-layer column
    x = ResidualBlocks(x, 3, 64, strides=2, name='conv2')
    x = ResidualBlocks(x, 4, 128, strides=2, name='conv3')
    x = ResidualBlocks(x, 6, 256, strides=2, name='conv4')
    x = ResidualBlocks(x, 3, 512, strides=1, name='conv5')

    x = layers.BatchNormalization(axis=3, name='post_bn')(x)
    outputs = layers.Activation('relu', name='post_relu')(x)

    model = Model(inputs, outputs, name='feature_extractor')
    return model


def ResidualBlocks(inputs, num_blocks, num_filters, strides, name):
    x = BottleneckBlock(
        inputs, num_filters, name=name + '_block1', strides=1, downsample=True)
    for i in range(2, num_blocks):
        x = BottleneckBlock(x, num_filters, name=name + '_block' + str(i))
    x = BottleneckBlock(
        x,
        num_filters,
        name=name + '_block' + str(num_blocks),
        strides=strides,
    )
    return x


def BottleneckBlock(
        inputs,
        num_filters,
        name,
        strides=1,
        downsample=False,
):
    # pre-activation
    preactivation = layers.BatchNormalization(
        axis=3, name=name + '_preact_bn')(inputs)
    preactivation = layers.Activation(
        'relu', name=name + '_preact_relu')(preactivation)

    # for the first block in a group, if it needs downsample, we need to do a projection
    # to shrink the dimension
    if downsample:
        identity = layers.Conv2D(
            num_filters * 4,  # lift channels first
            1,
            strides=strides,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(weight_decay),
            name=name + '_0_conv',
        )(preactivation)
    else:
        identity = layers.MaxPooling2D(
            1, strides=strides)(inputs) if strides > 1 else inputs

    # According to Figure 5 Right in [1]
    # 1x1 reduce dimension
    x = layers.Conv2D(
        num_filters,  # shrink channels as bottleneck designed
        1,
        strides=1,
        # with batch norm below, there's no need to use bias here
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name + '_1_conv',
    )(preactivation)
    # tensorflow is channel-last, so bach norm axis = 3
    x = layers.BatchNormalization(axis=3, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    # 3x3
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(
        num_filters,  # continue transforming in low channels
        3,
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(axis=3, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    # 1x1 increase dimension
    x = layers.Conv2D(
        num_filters * 4,  #  lift back to high channels again after bottleneck
        1,
        strides=1,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=name + '_3_conv',
    )(x)

    # add identity to outputd
    x = layers.Add(name=name + '_out')([identity, x])

    return x


def load_model_weights():
    base_path = ('https://github.com/keras-team/keras-applications/'
                 'releases/download/resnet/')
    hashes = {
        'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                       'fac2f116257151a9d068a22e544a4917'),
    }

    model_name = 'resnet50v2'
    file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
    file_hash = hashes[model_name][1]
    weights_path = tf.keras.utils.get_file(
        file_name,
        base_path + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    return weights_path


def ResNet50V2(input_shape, num_classes, pretrain=True):
    base_model = FeatureExtractor(input_shape=input_shape)
    base_model.trainable = True

    if pretrain:
        weights_path = load_model_weights()
        base_model.load_weights(weights_path, by_name=True)

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes),
        Softmax(),
    ])

    return model