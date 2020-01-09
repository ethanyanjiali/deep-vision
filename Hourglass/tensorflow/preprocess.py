import tensorflow as tf


class Preprocessor(object):
    def __init__(self,
                 image_shape=(256, 256, 3),
                 heatmap_shape=(64, 64, 16),
                 is_train=False):
        self.is_train = is_train
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape

    def __call__(self, example):
        features = self.parse_tfexample(example)
        image = tf.io.decode_jpeg(features['image/encoded'])

        if self.is_train:
            image = tf.image.resize(image, self.image_shape[0:2])
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, self.image_shape[0:2])

        image = tf.cast(image, tf.float32) / 127.5 - 1
        heatmap = self.make_heatmap(features)

        return image, heatmap

    def generate_2d_guassian(self, height, width, y0, x0, sigma=1):
        """
        "The same technique as Tompson et al. is used for supervision. A MeanSquared Error (MSE) loss is
        applied comparing the predicted heatmap to a ground-truth heatmap consisting of a 2D gaussian
        (with standard deviation of 1 px) centered on the joint location."
        """
        x, y = tf.meshgrid(tf.range(0, height, 1), tf.range(0, width, 1), indexing='xy')
        return tf.math.exp(-(tf.square(x - x0) + tf.math.square(y - y0))/ tf.math.square(sigma))


    def make_heatmap(self, features):
        visibility = features['image/object/parts/v']
        x = features['image/object/parts/x']
        y = features['image/object/parts/y']
        height = features['image/height']
        width = features['image/width']
        num_heatmap = heatmap_shape[2]
        heatmap = tf.zeros(heatmap_shape)
        for i in range(self.num_heatmap):
            if visibility[i] != 0:
                # Since the standard deviation is 1px, we have to convert center to absolute pixel first
                x0 = tf.math.round(x[i] * width)
                y0 = tf.math.round(y[i] * height)
                heatmap[:, :, i] = self.generate_2d_guassian(64, 64, y0, x0)
        return heatmap

    def parse_tfexample(self, example_proto):
        image_feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/object/parts/x': tf.io.VarLenFeature(tf.float32),
            'image/object/parts/y': tf.io.VarLenFeature(tf.float32),
            'image/object/parts/v': tf.io.VarLenFeature(tf.int32),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(example_proto,
                                          image_feature_description)