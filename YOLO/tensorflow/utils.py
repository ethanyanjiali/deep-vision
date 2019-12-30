import tensorflow as tf


def xywh_to_x1x2y1y2(box):
    xy = box[..., 0:2]
    wh = box[..., 2:4]

    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2

    y_box = tf.concat([x1y1, x2y2], axis=-1)
    return y_box


def xywh_to_y1x1y2x2(box):
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]

    yx = tf.concat([y, x], axis=-1)
    hw = tf.concat([h, w], axis=-1)

    y1x1 = yx - hw / 2
    y2x2 = yx + hw / 2

    y_box = tf.concat([y1x1, y2x2], axis=-1)
    return y_box


def broadcast_iou(box1, box2):
    """
    calculate iou between one box1iction box and multiple box2 box in a broadcast way

    inputs:
    box1: a tensor full of boxes, eg. (3, 4)
    box2: another tensor full of boxes, eg. (3, 4)
    """

    # assert one dimension in order to mix match box1 and box2
    # eg:
    # box1 -> (3, 1, 4)
    # box2 -> (1, 3, 4)
    box1 = tf.expand_dims(box1, -2)
    box2 = tf.expand_dims(box2, 0)

    # derive the union of shape to broadcast
    # eg. new_shape -> (3, 3, 4)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box1), tf.shape(box2))

    # broadcast (duplicate) box1 and box2 so that
    # each box2 has one box1 matched correspondingly
    # box1: (3, 3, 4)
    # box2: (3, 3, 4)
    box1 = tf.broadcast_to(box1, new_shape)
    box2 = tf.broadcast_to(box2, new_shape)

    # minimum xmax - maximum xmin is the width of intersection.
    # but has to be greater or equal to 0
    interserction_w = tf.maximum(
        tf.minimum(box1[..., 2], box2[..., 2]) - tf.maximum(
            box1[..., 0], box2[..., 0]), 0)
    # minimum ymax - maximum ymin is the height of intersection.
    # but has to be greater or equal to 0
    interserction_h = tf.maximum(
        tf.minimum(box1[..., 3], box2[..., 3]) - tf.maximum(
            box1[..., 1], box2[..., 1]), 0)
    intersection_area = interserction_w * interserction_h
    box1_area = (box1[..., 2] - box1[..., 0]) * \
        (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * \
        (box2[..., 3] - box2[..., 1])
    # intersection over union
    return intersection_area / (box1_area + box2_area - intersection_area)


def binary_cross_entropy(logits, labels):
    epsilon = 1e-7
    logits = tf.clip_by_value(logits, epsilon, 1 - epsilon)
    return -(labels * tf.math.log(logits) +
             (1 - labels) * tf.math.log(1 - logits))
