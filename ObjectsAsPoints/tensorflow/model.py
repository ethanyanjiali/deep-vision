import tensorflow as tf

from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Input,
    ReLU,
    MaxPool2D,
    UpSampling2D,
    BatchNormalization,
)

# [1] Stacked Hourglass Networks for Human Pose Estimation
# [2] Objects as Points

# https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L286
order_to_filters = {
    5: (256, 256),
    4: (256, 384),
    3: (384, 384),
    2: (384, 384),
    1: (384, 512),
}

# https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L287
order_to_num_residual = {
    5: 1,
    4: 1
    3: 1,
    2: 1,
    1: 3,
}


def ResidualBlock(inputs, filters, strides=1, downsample=False, name=None):
    """
    Please note that the residual block in ObjectsAsPoints is different from original HG-104
    https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L48
    """
    identity = inputs
    if downsample:
        identity = Conv2D(
            filters=filters,  # lift channels first
            kernel_size=1,
            strides=strides,
            padding='same',
            use_bias=False)(inputs)
        identity = BatchNormalization()(identity)

    x = Conv2D(
        filters=filters,
        kernel_size=1,
        strides=strides,
        padding='same',
        use_bias=False)(x)
    x = BatchNormalization()(inputs)
    x = ReLU()(x)

    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Add()([identity, x])
    x = ReLU()(x)

    return x


def DetectionConv(inputs, filters):
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(inputs)
    # There's no BN for detection conv
    # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L107
    x = ReLU()(y1)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    return x


def DetectionHead(inputs, num_classes):
    # heatmap for classes
    y1 = DetectionConv(inputs, num_classes)

    # size w, h
    y2 = DetectionConv(inputs, 2)

    # offset x, y
    y3 = DetectionConv(inputs, 2)

    return (y1, y2, y3)


def HourglassModule(inputs, order):
    """
    https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua#L3
    """
    filters1, filters2 = order_to_filters[order]
    num_residual = order_to_num_residual[order]

    # Upper branch
    up1 = BottleneckBlock(inputs, filters1, downsample=False)

    for i in range(num_residual):
        up1 = BottleneckBlock(up1, filters1, downsample=False)

    # Lower branch
    low1 = MaxPool2D(pool_size=2, strides=2)(inputs)
    for i in range(num_residual):
        low1 = BottleneckBlock(low1, filters2, downsample=False)

    low2 = low1
    if order > 1:
        low2 = HourglassModule(low1, order - 1, num_residual)
    else:
        for i in range(num_residual):
            low2 = BottleneckBlock(low2, filters2, downsample=False)

    low3 = low2
    for i in range(num_residual):
        low3 = BottleneckBlock(low3, filters1, downsample=False)

    up2 = UpSampling2D(size=2)(low3)

    out = Add()([up1, up2])

    return out


def ObjectsAsPoints(
        input_shape=(256, 256, 3), num_stack=4, num_residual=1,
        num_classes=80):
    """
    This part is significantly different from original HG-104
    Please refer to https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L253
    """
    inputs = Input(shape=input_shape)

    # initial processing of the image
    # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L195
    x = Conv2D(
        filters=128, kernel_size=7, strides=2, padding='same',
        use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = BottleneckBlock(x, 256, strides=2, downsample=True)

    intermediate = x
    ys = []

    for i in range(num_stack):

        x = HourglassModule(intermediate, order=5, num_residual=num_residual)
        x = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        y1, y2, y3 = DetectionHead(x, num_classes)
        ys.append((y1, y2, y3))

        # if it's not the last stack, we need to add predictions back
        if i < num_stack - 1:
            # although the original HG-104 doesn't have BN and ReLU for intermediate output
            # ObjectsAsPoints does have it
            # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L220
            x1 = Conv2D(filters=256, kernel_size=1, strides=1)(x)
            x1 = BatchNormalization()(x1)

            x2 = Conv2D(filters=256, kernel_size=1, strides=1)(intermediate)
            x2 = BatchNormalization()(x2)

            x = Add()([x1, x2])
            x = ReLU()(x)
            x = BottleneckBlock(x, 256, downsample=False)

    return tf.keras.Model(inputs, ys, name='stacked_hourglass')
