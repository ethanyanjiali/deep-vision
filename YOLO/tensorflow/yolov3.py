import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from utils import xywh_to_x1x2y1y2, xywh_to_y1x1y2x2, broadcast_iou, binary_cross_entropy

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416


def DarknetConv(inputs, filters, kernel_size, strides, name):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        name=name + '_conv2d',
        use_bias=False,
        # kernel_regularizer=tf.keras.regularizers.l2(0.0005)
    )(inputs)
    # YoloV2:
    # "By adding batch normalization on all of the convolutional layers in
    #  YOLO we get more than 2% improvement in mAP."
    x = BatchNormalization(name=name + '_bn')(x)
    # YoloV1:
    # "We use a linear activation function for the ﬁnal layer and all other
    #  layers use the following leaky rectiﬁed linear activation"
    x = LeakyReLU(alpha=0.1, name=name + '_leakyrelu')(x)
    return x


def DarknetResidual(inputs, filters1, filters2, name):
    shortcut = inputs
    x = DarknetConv(
        inputs, filters=filters1, kernel_size=1, strides=1, name=name + '_1x1')
    x = DarknetConv(
        x, filters=filters2, kernel_size=3, strides=1, name=name + '_3x3')
    x = Add(name=name + '_add')([shortcut, x])
    return x


def Darknet(shape=(256, 256, 3)):
    # YoloV3:
    # Table 1. Darknet-53.
    inputs = Input(shape=shape)

    x = DarknetConv(inputs, 32, kernel_size=3, strides=1, name='conv2d_0')

    x = DarknetConv(x, 64, kernel_size=3, strides=2, name='conv2d_1')
    # 1x residual blocks
    for i in range(1):
        x = DarknetResidual(x, 32, 64, 'residual_0_' + str(i))

    x = DarknetConv(x, 128, kernel_size=3, strides=2, name='conv2d_2')
    # 2x residual blocks
    for i in range(2):
        x = DarknetResidual(x, 64, 128, 'residual_1_' + str(i))

    x = DarknetConv(x, 256, kernel_size=3, strides=2, name='conv2d_3')
    # 8x residual blocks
    for i in range(8):
        x = DarknetResidual(x, 128, 256, 'residual_2_' + str(i))

    y0 = x

    x = DarknetConv(x, 512, kernel_size=3, strides=2, name='conv2d_4')
    # 8x residual blocks
    for i in range(8):
        x = DarknetResidual(x, 256, 512, 'residual_3_' + str(i))

    y1 = x

    x = DarknetConv(x, 1024, kernel_size=3, strides=2, name='conv2d_5')
    # 4x residual blocks
    for i in range(4):
        x = DarknetResidual(x, 512, 1024, 'residual_4_' + str(i))

    y2 = x

    return tf.keras.Model(inputs, (y0, y1, y2), name='darknet_53')


def YoloV3(shape=(416, 416, 3), num_classes=2, training=False):
    # YoloV3:
    # "In our experiments with COCO [10] we predict 3 boxes at each scale so
    #  the tensor is N × N × [3 ∗ (4 + 1 + 80)] for the 4 bounding box offsets,
    #  1 objectness prediction, and 80 class predictions."
    # 3 * (4 + 1 + num_classes) = 21
    final_filters = 3 * (4 + 1 + num_classes)

    inputs = Input(shape=shape)

    backbone = Darknet(shape)
    x_small, x_medium, x_large = backbone(inputs)

    # large scale detection
    # https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/cfg/yolov3.cfg#L549-L788
    x = DarknetConv(
        x_large,
        512,
        kernel_size=1,
        strides=1,
        name='detector_scale_large_1x1_1')
    x = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_1')
    x = DarknetConv(
        x, 512, kernel_size=1, strides=1, name='detector_scale_large_1x1_2')
    x = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_2')
    x = DarknetConv(
        x, 512, kernel_size=1, strides=1, name='detector_scale_large_1x1_3')

    y_large = DarknetConv(
        x, 1024, kernel_size=3, strides=1, name='detector_scale_large_3x3_3')
    y_large = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_large_final_conv2d',
    )(y_large)

    # meidum scale detection
    # YoloV3:
    # "Next we take the feature map from 2 layers previous and upsample it by 2×. We also take a feature map from earlier
    #  in the network and merge it with our upsampled features using concatenation. This method allows us to get more
    #  meaningful semantic information from the upsampled features and ﬁner-grained information from the earlier feature map.
    #  We then add a few more convolutional layers to process this combined feature map, and eventually predict a similar
    #  tensor, although now twice the size."
    #
    # From the code, 1x1x256 -> upsampling by 2 -> 3 times (1x1x256 -> 3x3x512)
    # https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/cfg/yolov3.cfg#L621
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_0')

    # Although not explained in the paper, the upsampling mentioned by the author
    # is just interprolation as seen from here
    # https://github.com/pjreddie/darknet/blob/61c9d02ec461e30d55762ec7669d6a1d3c356fb2/src/blas.c#L334
    x = UpSampling2D(size=(2, 2), name='detector_scale_1_upsampling')(x)
    x = Concatenate(name='detector_scale_1_concat')([x, x_medium])

    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_1')
    x = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_1')
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_2')
    x = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_2')
    x = DarknetConv(
        x, 256, kernel_size=1, strides=1, name='detector_scale_medium_1x1_3')

    y_medium = DarknetConv(
        x, 512, kernel_size=3, strides=1, name='detector_scale_medium_3x3_3')
    y_medium = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_medium_final_conv2d',
    )(y_medium)

    # small scale detection
    # YoloV3:
    # "We perform the same design one more time to predict boxes for the ﬁnal scale."
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_0')
    x = UpSampling2D(size=(2, 2), name='detector_scale_small_upsampling')(x)
    x = Concatenate(name='detector_scale_small_concat')([x, x_small])

    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_1')
    x = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_1')
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_2')
    x = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_2')
    x = DarknetConv(
        x, 128, kernel_size=1, strides=1, name='detector_scale_small_1x1_3')

    y_small = DarknetConv(
        x, 256, kernel_size=3, strides=1, name='detector_scale_small_3x3_3')
    y_small = Conv2D(
        filters=final_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        name='detector_scale_small_final_conv2d',
    )(y_small)

    # reshape (N, grid, grid, 21) into (N, grid, grid, 3, 7) to seprate predictions
    # for each anchor
    y_small_shape = tf.shape(y_small)
    y_medium_shape = tf.shape(y_medium)
    y_large_shape = tf.shape(y_large)

    y_small = tf.reshape(
        y_small, (y_small_shape[0], y_small_shape[1], y_small_shape[2], 3, -1),
        name='detector_reshape_small')
    y_medium = tf.reshape(
        y_medium,
        (y_medium_shape[0], y_medium_shape[1], y_medium_shape[2], 3, -1),
        name='detector_reshape_meidum')
    y_large = tf.reshape(
        y_large, (y_large_shape[0], y_large_shape[1], y_large_shape[2], 3, -1),
        name='detector_reshape_large')

    if training:
        return tf.keras.Model(inputs, (y_small, y_medium, y_large))

    box_small = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[0:3], num_classes),
        name='detector_final_box_small')(y_small)
    box_medium = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[3:6], num_classes),
        name='detector_final_box_medium')(y_medium)
    box_large = Lambda(
        lambda x: get_absolute_yolo_box(x, anchors_wh[6:9], num_classes),
        name='detector_final_box_large')(y_large)

    outputs = (box_small, box_medium, box_large)
    return tf.keras.Model(inputs, outputs)


def get_absolute_yolo_box(y_pred, valid_anchors_wh, num_classes):
    """
    Given a cell offset prediction from the model, calculate the absolute box coordinates to the whole image.
    It's also an adpation of the original C code here:
    https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c#L83
    note that, we divide w and h by grid size 

    inputs:
    y_pred: Prediction tensor from the model output, in the shape of (batch, grid, grid, anchor, 5 + num_classes)

    outputs:
    y_box: boxes in shape of (batch, grid, grid, anchor, 4), the last dimension is (xmin, ymin, xmax, ymax)
    objectness: probability that an object exists
    classes: probability of classes
    """

    t_xy, t_wh, objectness, classes = tf.split(
        y_pred, (2, 2, 1, num_classes), axis=-1)

    objectness = tf.sigmoid(objectness)
    classes = tf.sigmoid(classes)

    grid_size = tf.shape(y_pred)[1]
    # meshgrid generates a grid that repeats by given range. It's the Cx and Cy in YoloV3 paper.
    # for example, tf.meshgrid(tf.range(3), tf.range(3)) will generate a list with two elements
    # note that in real code, the grid_size should be something like 13, 26, 52 for examples here and below
    #
    # [[0, 1, 2],
    #  [0, 1, 2],
    #  [0, 1, 2]]
    #
    # [[0, 0, 0],
    #  [1, 1, 1],
    #  [2, 2, 2]]
    #
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))

    # next, we stack two items in the list together in the last dimension, so that
    # we can interleve these elements together and become this:
    #
    # [[[0, 0], [1, 0], [2, 0]],
    #  [[0, 1], [1, 1], [2, 1]],
    #  [[0, 2], [1, 2], [2, 2]]]
    #
    C_xy = tf.stack(C_xy, axis=-1)

    # let's add an empty dimension at axis=2 to expand the tensor to this:
    #
    # [[[[0, 0]], [[1, 0]], [[2, 0]]],
    #  [[[0, 1]], [[1, 1]], [[2, 1]]],
    #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
    #
    # at this moment, we now have a grid, which can always give us (y, x)
    # if we access grid[x][y]. For example, grid[0][1] == [[1, 0]]
    C_xy = tf.expand_dims(C_xy, axis=2)  # [gx, gy, 1, 2]

    # YoloV2, YoloV3:
    # bx = sigmoid(tx) + Cx
    # by = sigmoid(ty) + Cy
    #
    # for example, if all elements in b_xy are (0.1, 0.2), the result will be
    #
    # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
    #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
    #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]
    #
    b_xy = tf.sigmoid(t_xy) + tf.cast(C_xy, tf.float32)

    # finally, divide this absolute box_xy by grid_size, and then we will get the normalized bbox centroids
    # for each anchor in each grid cell. b_xy is now in shape (batch_size, grid_size, grid_size, num_anchor, 2)
    #
    # [[[[0.1/3, 0.2/3]], [[1.1/3, 0.2/3]], [[2.1/3, 0.2/3]]],
    #  [[[0.1/3, 1.2/3]], [[1.1/3, 1.2]/3], [[2.1/3, 1.2/3]]],
    #  [[[0.1/3, 2.2/3]], [[1.1/3, 2.2/3]], [[2.1/3, 2.2/3]]]]
    #
    b_xy = b_xy / tf.cast(grid_size, tf.float32)

    # YoloV2:
    # "If the cell is offset from the top left corner of the image by (cx , cy)
    # and the bounding box prior has width and height pw , ph , then the predictions correspond to: "
    #
    # https://github.com/pjreddie/darknet/issues/568#issuecomment-469600294
    # "It’s OK for the predicted box to be wider and/or taller than the original image, but
    # it does not make sense for the box to have a negative width or height. That’s why
    # we take the exponent of the predicted number."
    b_wh = tf.exp(t_wh) * valid_anchors_wh

    y_box = tf.concat([b_xy, b_wh], axis=-1)
    return y_box, objectness, classes


def get_relative_yolo_box(y_true, valid_anchors_wh):
    """
    This is the inverse of `get_absolute_yolo_box` above. It's turning (bx, by, bw, bh) into
    (tx, ty, tw, th) that is relative to cell location.
    """
    grid_size = tf.shape(y_true)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.expand_dims(tf.stack(C_xy, axis=-1), axis=2)

    b_xy = y_true[..., 0:2]
    b_wh = y_true[..., 2:4]
    t_xy = b_xy * tf.cast(grid_size, tf.float32) - tf.cast(C_xy, tf.float32)

    t_wh = tf.math.log(b_wh / valid_anchors_wh)
    # b_wh could have some cells are 0, divided by anchor could result in inf or nan
    t_wh = tf.where(
        tf.logical_or(tf.math.is_inf(t_wh), tf.math.is_nan(t_wh)),
        tf.zeros_like(t_wh), t_wh)

    y_box = tf.concat([t_xy, t_wh], axis=-1)
    return y_box


class YoloLoss(object):
    def __init__(self, num_classes, valid_anchors_wh):
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.valid_anchors_wh = valid_anchors_wh
        self.lambda_coord = 5.0
        self.lamda_noobj = 0.5

    def __call__(self, y_true, y_pred):
        """
        calculate the loss of model prediction for one scale
        """
        # for xy and wh, I seperated them into two groups with different suffix
        # suffix rel (relative) means that its coordinates are relative to cells
        # basically (tx, ty, tw, th) format from the paper
        # _rel is used to calcuate the loss
        # suffix abs (absolute) means that its coordinates are absolute with in whole image
        # basically (bx, by, bw, bh) format from the paper
        # _abs is used to calcuate iou and ignore mask

        # split y_pred into xy, wh, objectness and one-hot classes
        # pred_xy_rel: (batch, grid, grid, anchor, 2)
        # pred_wh_rel: (batch, grid, grid, anchor, 2)
        # TODO: Add comment for the sigmoid here
        pred_xy_rel = tf.sigmoid(y_pred[..., 0:2])
        pred_wh_rel = y_pred[..., 2:4]

        # this box is used to calculate iou, NOT loss. so we can't use
        # cell offset anymore and have to transform it into true values
        # both pred_obj and pred_class has been sigmoid'ed here
        # pred_xy_abs: (batch, grid, grid, anchor, 2)
        # pred_wh_abs: (batch, grid, grid, anchor, 2)
        # pred_obj: (batch, grid, grid, anchor, 1)
        # pred_class: (batch, grid, grid, anchor, num_classes)
        pred_box_abs, pred_obj, pred_class = get_absolute_yolo_box(
            y_pred, self.valid_anchors_wh, self.num_classes)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)

        # split y_true into xy, wh, objectness and one-hot classes
        # pred_xy_abs: (batch, grid, grid, anchor, 2)
        # pred_wh_abs: (batch, grid, grid, anchor, 2)
        # pred_obj: (batch, grid, grid, anchor, 1)
        # pred_class: (batch, grid, grid, anchor, num_classes)
        true_xy_abs, true_wh_abs, true_obj, true_class = tf.split(
            y_true, (2, 2, 1, self.num_classes), axis=-1)
        true_box_abs = tf.concat([true_xy_abs, true_wh_abs], axis=-1)
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)

        # true_box_rel: (batch, grid, grid, anchor, 4)
        true_box_rel = get_relative_yolo_box(y_true, self.valid_anchors_wh)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        # some adjustment to improve small box detection, note the (2-truth.w*truth.h) below
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c#L190
        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        # YoloV2:
        # "If the cell is offset from the top left corner of the image by (cx , cy)
        # and the bounding box prior has width and height pw , ph , then the predictions correspond to:"
        #
        # to calculate the iou and determine the ignore mask, we need to first transform
        # prediction into real coordinates (bx, by, bw, bh)

        # YoloV2:
        # "This ground truth value can be easily computed by inverting the equations above."
        #
        # to calculate loss and differentiation, we need to transform ground truth into
        # cell offset first like demonstrated here:
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c#L93
        xy_loss = self.calc_xy_loss(true_obj, true_xy_rel, pred_xy_rel, weight)
        wh_loss = self.calc_wh_loss(true_obj, true_wh_rel, pred_wh_rel, weight)
        class_loss = self.calc_class_loss(true_obj, true_class, pred_class)

        # use the absolute yolo box to calculate iou and ignore mask
        ignore_mask = self.calc_ignore_mask(true_obj, true_box_abs,
                                            pred_box_abs)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)

        # YoloV1: Function (3)
        return xy_loss + wh_loss + class_loss + obj_loss, (xy_loss, wh_loss,
                                                           class_loss,
                                                           obj_loss)

    def calc_ignore_mask(self, true_obj, true_box, pred_box):
        # eg. true_obj (1, 13, 13, 3, 1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        # eg. true_obj (1, 13, 13, 3)
        # eg. true_box (1, 13, 13, 3, 4)
        # eg. pred_box (1, 13, 13, 2, 4)
        # eg. true_box_filtered (2, 4) it was (3, 4) but one element got filtered out
        true_box_filtered = tf.boolean_mask(true_box, tf.cast(
            true_obj, tf.bool))

        # YOLOv3:
        # "If the bounding box prior is not the best but does overlap a ground
        # truth object by more than some threshold we ignore the prediction,
        # following [17]. We use the threshold of .5."
        # calculate the iou for each pair of pred bbox and true bbox, then find the best among them
        # eg. best_iou (1, 1, 1, 2)
        best_iou = tf.reduce_max(
            broadcast_iou(pred_box, true_box_filtered), axis=-1)

        # if best iou is higher than threshold, set the box to be ignored for noobj loss
        # eg. ignore_mask(1, 1, 1, 2)
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):
        """
        calculate loss of objectness: sum of L2 distances

        inputs:
        true_obj: objectness from ground truth in shape of (batch, grid, grid, anchor, num_classes)
        pred_obj: objectness from model prediction in shape of (batch, grid, grid, anchor, num_classes)

        outputs:
        obj_loss: objectness loss
        """
        obj_entropy = binary_cross_entropy(pred_obj, true_obj)

        obj_loss = true_obj * obj_entropy
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        noobj_loss = tf.reduce_sum(
            noobj_loss, axis=(1, 2, 3, 4)) * self.lamda_noobj

        return obj_loss + noobj_loss

    def calc_class_loss(self, true_obj, true_class, pred_class):
        """
        calculate loss of class prediction

        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_class: one-hot class from ground truth in shape of (batch, grid, grid, anchor, num_classes)
        pred_class: one-hot class from model prediction in shape of (batch, grid, grid, anchor, num_classes)

        outputs:
        class_loss: class loss
        """
        # Yolov1:
        # "Note that the loss function only penalizes classiﬁcation error
        # if an object is present in that grid cell (hence the conditional
        # class probability discussed earlier).
        class_loss = binary_cross_entropy(pred_class, true_class)
        class_loss = true_obj * class_loss
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))
        return class_loss

    def calc_xy_loss(self, true_obj, true_xy, pred_xy, weight):
        """
        calculate loss of the centroid coordinate: sum of L2 distances

        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_xy: centroid x and y from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_xy: centroid x and y from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box

        outputs:
        xy_loss: centroid loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)

        # in order to element-wise multiply the result from tf.reduce_sum
        # we need to squeeze one dimension for objectness here
        true_obj = tf.squeeze(true_obj, axis=-1)

        # YoloV1:
        # "It also only penalizes bounding box coordinate error if that
        # predictor is "responsible" for the ground truth box (i.e. has the
        # highest IOU of any predictor in that grid cell)."
        xy_loss = true_obj * xy_loss * weight

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * self.lambda_coord

        return xy_loss

    def calc_wh_loss(self, true_obj, true_wh, pred_wh, weight):
        """
        calculate loss of the width and height: sum of L2 distances

        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_wh: width and height from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_wh: width and height from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box

        outputs:
        wh_loss: width and height loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        wh_loss = true_obj * wh_loss * weight
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3)) * self.lambda_coord
        return wh_loss
