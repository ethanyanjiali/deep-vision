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


def broadcast_iou(box_a, box_b):
    """
    calculate iou between box_a and multiple box_b in a broadcast way.
    Used this implementation as reference: 
    https://github.com/dmlc/gluon-cv/blob/c3dd20d4b1c1ef8b7d381ad2a7d04a68c5fa1221/gluoncv/nn/bbox.py#L206

    inputs:
    box_a: a tensor full of boxes, eg. (B, N, 4), box is in x1y1x2y2
    box_b: another tensor full of boxes, eg. (B, M, 4)
    """

    # (B, N, 1, 4)
    box_a = tf.expand_dims(box_a, -2)
    # (B, 1, M, 4)
    box_b = tf.expand_dims(box_b, -3)
    # (B, N, M, 4)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_a), tf.shape(box_b))

    # (B, N, M, 4)
    # (B, N, M, 4)
    box_a = tf.broadcast_to(box_a, new_shape)
    box_b = tf.broadcast_to(box_b, new_shape)

    # (B, N, M, 1)
    al, at, ar, ab = tf.split(box_a, 4, -1)
    bl, bt, br, bb = tf.split(box_b, 4, -1)

    # (B, N, M, 1)
    left = tf.math.maximum(al, bl)
    right = tf.math.minimum(ar, br)
    top = tf.math.maximum(at, bt)
    bot = tf.math.minimum(ab, bb)

    # (B, N, M, 1)
    iw = tf.clip_by_value(right - left, 0, 1)
    ih = tf.clip_by_value(bot - top, 0, 1)
    i = iw * ih

    # (B, N, M, 1)
    area_a = (ar - al) * (ab - at)
    area_b = (br - bl) * (bb - bt)
    union = area_a + area_b - i

    # (B, N, M)
    iou = tf.squeeze(i / (union + 1e-7), axis=-1)

    return iou


def binary_cross_entropy(logits, labels):
    epsilon = 1e-7
    logits = tf.clip_by_value(logits, epsilon, 1 - epsilon)
    return -(labels * tf.math.log(logits) +
             (1 - labels) * tf.math.log(1 - logits))
