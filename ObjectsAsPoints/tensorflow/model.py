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
    5: (2, 2),
    4: (2, 2),
    3: (2, 2),
    2: (2, 2),
    1: (2, 4),
}


def ResidualBlock(inputs, filters_in, filters_out, strides=1, name=None):
    """
    Please note that the residual block in ObjectsAsPoints is different from original HG
    https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L48
    """
    identity = inputs
    if filters_in != filters_out or strides > 1:
        identity = Conv2D(
            filters=filters_out,  # lift channels first
            kernel_size=1,
            strides=strides,
            padding='same',
            use_bias=False)(inputs)
        identity = BatchNormalization()(identity)

    x = Conv2D(
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        padding='same',
        use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(
        filters=filters_out,
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
    x = ReLU()(x)
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
    https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py#L117
    """
    curr_filters, next_filters = order_to_filters[order]
    curr_residual, next_residual = order_to_num_residual[order]

    # Upper branch
    up1 = ResidualBlock(inputs, curr_filters, curr_filters)
    for i in range(curr_residual - 1):
        up1 = ResidualBlock(up1, curr_filters, curr_filters)

    # Lower branch
    low1 = ResidualBlock(inputs, curr_filters, next_filters, strides=2)
    for i in range(curr_residual - 1):
        low1 = ResidualBlock(low1, next_filters, next_filters)

    low2 = low1
    if order > 1:
        low2 = HourglassModule(low2, order - 1)
    else:
        for i in range(next_residual):
            low2 = ResidualBlock(low2, next_filters, next_filters)
    
    low3 = low2
    for i in range(curr_residual - 1):
        low3 = ResidualBlock(low3, next_filters, next_filters)
    low3 = ResidualBlock(low2, next_filters, curr_filters)

    up2 = UpSampling2D(size=2)(low3)

    out = Add()([up1, up2])

    return out


def ObjectsAsPoints(
        input_shape=(256, 256, 3), num_stack=2, num_classes=80):
    """
    This part is significantly different from original HG
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
    x = ResidualBlock(x, 128, 256, strides=2)

    intermediate = x
    ys = []

    for i in range(num_stack):
        x = HourglassModule(intermediate, order=5)
        x = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same')(x)
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

            intermediate = Add()([x1, x2])
            intermediate = ReLU()(intermediate)
            intermediate = ResidualBlock(x, 256, 256)


    return tf.keras.Model(inputs, ys, name='objects_as_points')
