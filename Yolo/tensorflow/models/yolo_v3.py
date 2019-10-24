import tensorflow as tf
import numpy as np

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
from tensorflow.keras import Model
from tensorflow.keras.losses import (binary_crossentropy,
                                     sparse_categorical_crossentropy)

yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198),
                         (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(
        filters=filters,
        kernel_size=size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


# def DarknetConv(inputs, filters, kernel_size, strides=1):
#     x = Conv2D(
#         filters=filters,
#         kernel_size=kernel_size,
#         strides=strides,
#         padding='same',
# #         kernel_regularizer=tf.keras.regularizers.l2(0.0005)
#     )(inputs)
#     x = BatchNormalization()(x)
#     x = LeakyReLU(alpha=0.1)(x)
#     return x

# def DarknetResidual(inputs, filters1, filters2):
#     shortcut = inputs
#     x = DarknetConv(inputs, filters=filters1, kernel_size=1)
#     x = DarknetConv(x, filters=filters2, kernel_size=3)
#     x = Add()([shortcut, x])
#     return x

# def Darknet(name):
#     # Table 1. Darknet-53.
#     inputs = Input([None, None, 3])

#     x = DarknetConv(inputs, 32, kernel_size=3)

#     x = DarknetConv(x, 64, kernel_size=3, strides=2)
#     # 1x residual blocks
#     for _ in range(1):
#         x = DarknetResidual(x, 32, 64)

#     x = DarknetConv(inputs, 128, kernel_size=3, strides=2)
#     # 2x residual blocks
#     for _ in range(2):
#         x = DarknetResidual(x, 64, 128)

#     x = DarknetConv(inputs, 256, kernel_size=3, strides=2)
#     # 8x residual blocks
#     for _ in range(8):
#         x = DarknetResidual(x, 128, 256)

#     y1 = x

#     x = DarknetConv(inputs, 512, kernel_size=3, strides=2)
#     # 8x residual blocks
#     for _ in range(8):
#         x = DarknetResidual(x, 256, 512)

#     y2 = x

#     x = DarknetConv(inputs, 1024, kernel_size=3, strides=2)
#     # 4x residual blocks
#     for _ in range(4):
#         x = DarknetResidual(x, 512, 1024)

#     y3 = x

#     return tf.keras.Model(inputs, (y1, y2, y3))


def YoloConv(filters, name=None):

    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, num_anchors, classes, name=None):

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        # conv layer to use num_anchors * (classes + 5) filters, kernel 1x1
        x = DarknetConv(x, num_anchors * (classes + 5), 1)
        # reshape the output to be like (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        # `+5` here stands of x, y, w, h, obj
        x = Lambda(
            lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], num_anchors, classes + 5))
        )(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def yolo_boxes(pred, anchors, classes):
    """Transform the model ouput into final logits and readable bounding box format

    Args:
        pred: the prediction from conv layers directly in the format of 
            (batch_size, grid, grid, num_anchors, (x, y, w, h, obj, ...classes))
        anchors: the x and y multiplier used to map normalized prediction into pixel values.
            For example, if anchors is (116, 90), prediction is (0.1, 0.2), then the
            pixel position would be (11.6, 1.8)
        classes: total number of classes we are predicting
    Returns:
        A quadruple result in the follow format 
            (
                minmax_bbox,
                objectness,
                class_prob,
                xyhw_bbox,
            )
        bbox: common bbox coordinates (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax) in shape of
            (batch_size, grid_size, grid_size, num_anchor, 4)
        objectness: single float number of indicate if there's an object in shape of
            (batch_size, grid_size, grid_size, num_anchor, 1)
        class_prob: probability of each class in shape of
            (batch_size, grid_size, grid_size, num_anchor, num_classes)
        pred_box: original xywh prediction to calculate loss later. shape is
            (batch_size, grid_size, grid_size, num_anchor, 4)
    """
    # split the prediction into useful parts
    # shapes afer splitting will be like (batch_size, grid_size, grid_size, num_anchor, X)
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    # takes the sigmoid of different type of prediction seperately
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    # keeps the original xywh prediction
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    # meshgrid generates a grid that repeat by given range
    # for example, tf.meshgrid(tf.range(3), tf.range(3)) will generate a list with two elements:
    #
    # [[0, 1, 2],
    #  [0, 1, 2],
    #  [0, 1, 2]]
    #
    # [[0, 0, 0],
    #  [1, 1, 1],
    #  [2, 2, 2]]
    #
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))

    # next, we stack two items in the list together in the last dimension, so that
    # we can interleve these elements together and become this:
    #
    # [[[0, 0], [1, 0], [2, 0]],
    #  [[0, 1], [1, 1], [2, 1]],
    #  [[0, 2], [1, 2], [2, 2]]]
    #
    grid = tf.stack(grid, axis=-1)

    # let's add an empty dimension at axis=2 to expand the tensor to this:
    #
    # [[[[0, 0]], [[1, 0]], [[2, 0]]],
    #  [[[0, 1]], [[1, 1]], [[2, 1]]],
    #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
    #
    # at this moment, we now have a grid, which can always give us (y, x)
    # if we access grid[x][y]. For example, grid[0][1] == [[1, 0]]
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    # add bounding box xy to the grid, for example, if all elements in bbox_xy are (0.1, 0.2), the result will be
    #
    # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
    #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
    #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]
    #
    box_xy = (box_xy + tf.cast(grid, tf.float32))
    # finally, divide this grid by grid_size, and then we will get the normalized bbox centroids
    # for each anchor in each grid cell. bbox_xy is now in shape (batch_size, grid_size, grid_size, num_anchor, 2)
    #
    # [[[[0.1/3, 0.2/3]], [[1.1/3, 0.2/3]], [[2.1/3, 0.2/3]]],
    #  [[[0.1/3, 1.2/3]], [[1.1/3, 1.2]/3], [[2.1/3, 1.2/3]]],
    #  [[[0.1/3, 2.2/3]], [[1.1/3, 2.2/3]], [[2.1/3, 2.2/3]]]]
    #
    box_xy = box_xy / tf.cast(grid_size, tf.float32)

    # https://github.com/pjreddie/darknet/issues/568#issuecomment-469600294
    # It’s OK for the predicted box to be wider and/or taller than the original image, but
    # it does not make sense for the box to have a negative width or height. That’s why
    # we take the exponent of the predicted number.
    # We multiply by anchors here because each grid covers anchors percentage of pixels in
    # the input image
    box_wh = tf.exp(box_wh) * anchors

    # add or substract half of width and half of height to the bbox centroid to
    # get the min coordinates and max coordinates pf bbox individually
    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs):
    """Performs non max suppression over yolo bbox predictions

    Args:
        outputs: the prediction from three conv outputs (3, (minmax_bbox,objectness,class_prob,xyhw_bbox))
    Returns:
        'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
            containing the non-max suppressed boxes.
        'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
            the scores for the boxes.
        'nmsed_classes': A [batch_size, max_detections] float32 tensor
            containing the class for boxes.
        'valid_detections': A [batch_size] int32 tensor indicating the number of
            valid detections per batch item. Only the top valid_detections[i] entries
            in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
            entries are zero paddings.
    """
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        # each output will be like (minmax_bbox,objectness,class_prob,xyhw_bbox)
        # o[0] is boxes. This flatten bboxes to (batch_size, num grids * achor per grid, 4)
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        # o[1] is objectness. This flatten objectness to (batch_size, num grids * achor per grid, 1)
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        # o[2] is class_prob. This flatten class_prob to (batch_size, num grids * achor per grid, num_classes)
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    # concat multiple outputs' flatten result together. for example, if b is a list of three [32, 100, 4]
    # then the combined result will be [32, 300, 4]
    bboxes = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    # calculate the score for each class by multiplying objectnes to class probability
    scores = confidence * class_probs

    # Reshape to meet `tf.image.combined_non_max_suppression` input requirements
    # bbox is A 4-D float `Tensor` of shape `[batch_size, num_boxes, q, 4]`. If `q`
    # is 1 then same boxes are used for all classes otherwise, if `q` is equal
    # to number of classes, class-specific boxes are used.
    # scores is a 3-D float `Tensor` of shape `[batch_size, num_boxes, num_classes]`
    # representing a single score corresponding to each box (each row of boxes).
    bboxes = tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 1, 4))
    scores = tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1]))

    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=bboxes,
        scores=scores,
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold)

    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def YoloV3(size=None,
           channels=3,
           anchors=yolo_anchors,
           masks=yolo_anchor_masks,
           classes=602,
           training=False):
    x = inputs = Input([size, size, channels])

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    # When training, return the raw model output like (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    # Otherwise, output human readable bounding boxes
    boxes_0 = Lambda(
        lambda x: yolo_boxes(x, anchors[masks[0]], classes),
        name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(
        lambda x: yolo_boxes(x, anchors[masks[1]], classes),
        name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(
        lambda x: yolo_boxes(x, anchors[masks[2]], classes),
        name='yolo_boxes_2')(output_2)

    outputs = Lambda(
        lambda x: yolo_nms(x), name='yolo_nms')((boxes_0[:3], boxes_1[:3],
                                                 boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloLoss(anchors, classes=602, ignore_thresh=0.5):

    def broadcast_iou(box_1, box_2):
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (N, (x1, y1, x2, y2))

        # broadcast boxes
        box_1 = tf.expand_dims(box_1, -2)
        box_2 = tf.expand_dims(box_2, 0)
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)

        int_w = tf.maximum(
            tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(
                box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(
            tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(
                box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
            (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
            (box_2[..., 3] - box_2[..., 1])
        return int_area / (box_1_area + box_2_area - int_area)

    def yolo_loss(y_true, y_pred):
        """
        y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        """
        # Transform all prediction into logits and boxes
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # Transform all ground truth into logits and boxes
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # Give higher weights to small boxes
        true_box_area = true_wh[..., 0] * true_wh[..., 1]
        box_loss_scale = 2 - true_box_area

        # Translate ground truth into xywh relative to each grid cell. We
        # do this by inverting the logic of the meshgrid in `yolo_boxes`. Take a
        # look of that function to understand what's going on here.
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        # invert the grid
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        # invert the exp and normalization
        true_wh = tf.math.log(true_wh / anchors)

        true_wh = tf.where(
            tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # squeeze into shape of (batch_size, grid, grid, anchor)
        obj_mask = tf.squeeze(true_obj, -1)
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        # calculate the iou for each pair of pred bbox and true bbox, then find the best among them
        best_iou = tf.reduce_max(
            broadcast_iou(pred_box, true_box_flat), axis=-1)
        # if best iou is lower than threshold, set this pred box to ignore
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # calculate loss of the centroid coordinate: sum of L2 distances
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        # calculate loss of the weight and height: sum of L2 distances
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        # calculate loss of objectness: binary_crossentropy
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return yolo_loss