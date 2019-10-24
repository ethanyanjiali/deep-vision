#!/usr/bin/env python
# coding: utf-8

# In[4]:

import tensorflow as tf
from models.yolo_v3 import Darknet, YoloV3, YoloConv, YoloLoss
import numpy as np
# import horovod.tensorflow as hvd

# In[2]:

num_classes = 602
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198),
                         (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# In[3]:


def parse_example_proto(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded':
        tf.io.FixedLenFeature((), dtype=tf.string, default_value=''),
        'image/class/label':
        tf.io.VarLenFeature(tf.int64),
        'image/class/text':
        tf.io.VarLenFeature(tf.string),
        'image/object/class/label':
        tf.io.VarLenFeature(tf.int64),
    }
    feature_map.update({
        k: tf.io.VarLenFeature(dtype=tf.float32)
        for k in [
            'image/object/bbox/xmin', 'image/object/bbox/ymin',
            'image/object/bbox/xmax', 'image/object/bbox/ymax'
        ]
    })
    features = tf.io.parse_single_example(example_serialized, feature_map)

    image_buffer = features['image/encoded']
    bboxes = tf.stack([
        tf.sparse.to_dense(features['image/object/bbox/xmin']),
        tf.sparse.to_dense(features['image/object/bbox/ymin']),
        tf.sparse.to_dense(features['image/object/bbox/xmax']),
        tf.sparse.to_dense(features['image/object/bbox/ymax']),
        tf.cast(
            tf.sparse.to_dense(features['image/object/class/label']),
            tf.float32)
    ],
                      axis=1)

    paddings = [[0, 800 - tf.shape(bboxes)[0]], [0, 0]]
    bboxes = tf.pad(bboxes, paddings)
    return image_buffer, bboxes


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):
    # y_true: (boxes, (x1, y1, x2, y2, class, best_anchor))
    # y_true_out: (grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros((grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for j in tf.range(tf.shape(y_true)[1]):
        if tf.equal(y_true[j][2], 0):
            continue
        anchor_eq = tf.equal(anchor_idxs, tf.cast(y_true[j][5], tf.int32))

        if tf.reduce_any(anchor_eq):
            box = y_true[j][0:4]
            box_xy = (y_true[j][0:2] + y_true[j][2:4]) / 2

            anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
            grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

            # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
            indexes = indexes.write(idx,
                                    [grid_xy[1], grid_xy[0], anchor_idx[0][0]])
            updates = updates.write(
                idx, [box[0], box[1], box[2], box[3], 1, y_true[j][4]])
            idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(),
                                       updates.stack())


def preprocess_bboxes(bboxes, anchors, anchor_masks, classes):
    """
    Args:
        bboxes: all bounding boxes belong to the image in shape of [num_boxes, (xmin, ymin, xmax, ymax, label)]
    """
    y_outs = []
    grid_size = 13

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = bboxes[..., 2:4] - bboxes[..., 0:2]  # shape = [N, 2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(
        box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    bboxes = tf.concat([bboxes, anchor_idx], axis=-1)
    for anchor_idxs in anchor_masks:
        y_outs.append(
            transform_targets_for_output(bboxes, grid_size, anchor_idxs,
                                         classes))
        grid_size *= 2

    return tuple(y_outs)


def preprocess_image(image_buffer, output_size):
    image = tf.io.decode_jpeg(image_buffer)
    image = tf.image.resize(image, output_size)
    image = image / 255.0
    return image


def preprocess_data(example):
    image_buffer, bboxes = parse_example_proto(example)

    image = preprocess_image(image_buffer, (416, 416))
    bboxes13, bboxes26, bboxes52 = preprocess_bboxes(
        bboxes, yolo_anchors, yolo_anchor_masks, num_classes)

    return image, (bboxes13, bboxes26, bboxes52)


# In[4]:

# strategy = tf.distribute.MirroredStrategy()
# print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# BATCH_SIZE_PER_REPLICA = 32
# GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = 8
# from preprocess import parse_record
label_descriptions = ['' for i in range(num_classes)]
with open('./labels.txt') as fp:
    line = fp.readline()
    while line:
        parts = line.strip().split('\t')
        label_descriptions[int(parts[0])] = parts[1]
        line = fp.readline()

data_dir = 'gs://perception-data_warehouse-tfrecords/openimages/boxable/bbox-602classes/train'
# dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files('{}/*'.format(data_dir), seed=1))
dataset = dataset.map(preprocess_data)
dataset = dataset.batch(GLOBAL_BATCH_SIZE)

# dist_dataset = strategy.experimental_distribute_dataset(dataset)
checkpoint_prefix = './checkpoints/ckpt'

# In[ ]:

model = YoloV3(training=True)
loss = [
    YoloLoss(yolo_anchors[mask], classes=num_classes)
    for mask in yolo_anchor_masks
]
optimizer = tf.keras.optimizers.Adam(lr=0.0001)

for epoch in range(1, 2):
    for batch, (image, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            outputs = model(image)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if batch % 100 == 0:
            print("epoch: {}, train batch: {}, total loss: {}".format(
                epoch, batch, total_loss.numpy()))
    print("epoch: {}, total loss: {}".format(epoch, batch, total_loss.numpy()))

# In[ ]:
