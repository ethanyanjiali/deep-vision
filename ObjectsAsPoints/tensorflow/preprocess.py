import tensorflow as tf


class Preprocessor(object):
    def __init__(self, is_train, num_classes, output_shape=(416, 416)):
        self.is_train = is_train
        self.num_classes = num_classes
        self.output_shape = output_shape

    def __call__(self, example):
        features = self.parse_tfexample(example)

        encoded = features['image/encoded']
        image = tf.io.decode_jpeg(encoded)
        image = tf.cast(image, tf.float32)

        classes, bboxes = self.parse_y_features(features)

        if self.is_train:
            image, bboxes = self.random_flip_image_and_label(image, bboxes)
            image, bboxes = self.random_crop_image_and_label(image, bboxes)

        image = tf.image.resize(image, self.output_shape)
        image = tf.cast(image, tf.float32) / 127.5 - 1

        return image, bboxes

    def random_flip_image_and_label(self, image, bboxes):
        """
        flip left and right for 50% of images
        """
        r = tf.random.uniform([1])
        if r < 0.5:
            image = tf.image.flip_left_right(image)
            xmin, ymin, xmax, ymax = tf.split(bboxes, [1, 1, 1, 1], -1)
            # note that we need to switch here
            xmin, xmax = 1 - xmax, 1 - xmin
            bboxes = tf.squeeze(
                tf.stack([xmin, ymin, xmax, ymax], axis=1), axis=-1)

        return image, bboxes

    def get_random_crop_delta(self, bboxes):
        """
        get a random crop which includes all bounding boxes. Since all bboxes here belong to one image,
        we can calcualte the minimum of all xmin and ymin, and the maximum of all xmax and ymax to get
        the an area that can include all boxes. the crop will be randomly picked between this area boundary and
        the boundary of the whole image.
        """
        min_xmin = tf.math.reduce_min(bboxes[..., 0])
        min_ymin = tf.math.reduce_min(bboxes[..., 1])
        max_xmax = tf.math.reduce_max(bboxes[..., 2])
        max_ymax = tf.math.reduce_max(bboxes[..., 3])

        # delta is the normalized margin from bboxes boundary the crop boundary
        # ____________________________________
        # |         ________________         |
        # |image    |crop ______   |         |
        # |<-DELTA->|     |bbox|   |<-DELTA->|
        # |         |     |____|   |         |
        # |         |______________|         |
        # |__________________________________|
        xmin_delta = tf.random.uniform([1], 0, min_xmin)
        ymin_delta = tf.random.uniform([1], 0, min_ymin)
        xmax_delta = tf.random.uniform([1], 0, 1 - max_xmax)
        ymax_delta = tf.random.uniform([1], 0, 1 - max_ymax)

        return xmin_delta, ymin_delta, xmax_delta, ymax_delta

    def random_crop_image_and_label(self, image, bboxes):
        """
        crop images randomly at 50% chance but preserve all bounding boxes. the crop is guaranteed to include
        all bounding boxes. 
        """
        r = tf.random.uniform([1])
        if r < 0.5:
            xmin_delta, ymin_delta, xmax_delta, ymax_delta = self.get_random_crop_delta(
                bboxes)

            xmin, ymin, xmax, ymax = tf.split(bboxes, [1, 1, 1, 1], -1)
            # before crop: |_0.1_|_0.1_|____________0.5___________|_0.1_|___0.2___|
            # after crop:  |_0.1_|____________0.5___________|_0.1_|
            # imagine old xmin is 0.2 (0.1+0.1), old xmax is 0.8 (0.1+0.1+0.5+0.1)
            # if we cut both left 0.1 (xmin_delta) and right 0.2 (xmax_delta)
            # the new xmin will be (0.2 - 0.1) / (1 - 0.1 - 0.2) = 1/7
            # the new xmax will be (0.8 - 0.1) / (1 - 0.1 - 0.2) = 6/7
            # same thing for y
            xmin = (xmin - xmin_delta) / (1 - xmin_delta - xmax_delta)
            ymin = (ymin - ymin_delta) / (1 - ymin_delta - ymax_delta)
            xmax = (xmax - xmin_delta) / (1 - xmin_delta - xmax_delta)
            ymax = (ymax - ymin_delta) / (1 - ymin_delta - ymax_delta)

            bboxes = tf.squeeze(
                tf.stack([xmin, ymin, xmax, ymax], axis=1), axis=-1)
            h = tf.cast(tf.shape(image)[0], dtype=tf.float32)
            w = tf.cast(tf.shape(image)[1], dtype=tf.float32)

            offset_height = tf.cast(ymin_delta[0] * h, dtype=tf.int32)
            offset_width = tf.cast(xmin_delta[0] * w, dtype=tf.int32)
            target_height = tf.cast(
                tf.math.ceil((1 - ymax_delta - ymin_delta)[0] * h),
                dtype=tf.int32)
            target_width = tf.cast(
                tf.math.ceil((1 - xmax_delta - xmin_delta)[0] * w),
                dtype=tf.int32)

            image = image[offset_height:offset_height +
                          target_height, offset_width:offset_width +
                          target_width, :]
        return image, bboxes

    def parse_y_features(self, features):
        classes = tf.sparse.to_dense(features['image/object/class/label'])
        classes = tf.one_hot(classes, self.num_classes)

        # tf.pad(classes, [[0, 100 - tf.shape(classes)[0]], []], 'CONSTANT')

        # bboxes shape (None, 4)
        bboxes = tf.stack([
            tf.sparse.to_dense(features['image/object/bbox/xmin']),
            tf.sparse.to_dense(features['image/object/bbox/ymin']),
            tf.sparse.to_dense(features['image/object/bbox/xmax']),
            tf.sparse.to_dense(features['image/object/bbox/ymax']),
        ],
                          axis=1)
        return classes, bboxes

    def generate_2d_guassian(self, height, width, y0, x0, sigma=1):
        """
        "The same technique as Tompson et al. is used for supervision. A MeanSquared Error (MSE) loss is
        applied comparing the predicted heatmap to a ground-truth heatmap consisting of a 2D gaussian
        (with standard deviation of 1 px) centered on the joint location."

        https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/img.lua#L204
        """
        heatmap = tf.zeros((height, width))
        return heatmap

        # this gaussian patch is 7x7, let's get four corners of it first
        xmin = x0 - 3 * sigma
        ymin = y0 - 3 * sigma
        xmax = x0 + 3 * sigma
        ymax = y0 + 3 * sigma
        # if the patch is out of image boundary we simply return nothing according to the source code
        if xmin >= width or ymin >= height or xmax < 0 or ymax < 0:
            return heatmap

        size = 6 * sigma + 1
        x, y = tf.meshgrid(
            tf.range(0, 6 * sigma + 1, 1),
            tf.range(0, 6 * sigma + 1, 1),
            indexing='xy')

        # the center of the gaussian patch should be 1
        center_x = size // 2
        center_y = size // 2

        # generate this 7x7 gaussian patch
        gaussian_patch = tf.cast(
            tf.math.exp(
                -(tf.square(x - center_x) + tf.math.square(y - center_y)) /
                (tf.math.square(sigma) * 2)),
            dtype=tf.float32)

        # part of the patch could be out of the boundary, so we need to determine the valid range
        # if xmin = -2, it means the 2 left-most columns are invalid, which is max(0, -(-2)) = 2
        patch_xmin = tf.math.maximum(0, -xmin)
        patch_ymin = tf.math.maximum(0, -ymin)
        # if xmin = 59, xmax = 66, but our output is 64x64, then we should discard 2 right-most columns
        # which is min(64, 66) - 59 = 5, and column 6 and 7 are discarded
        patch_xmax = tf.math.minimum(xmax, width) - xmin
        patch_ymax = tf.math.minimum(ymax, height) - ymin

        # also, we need to determine where to put this patch in the whole heatmap
        heatmap_xmin = tf.math.maximum(0, xmin)
        heatmap_ymin = tf.math.maximum(0, ymin)
        heatmap_xmax = tf.math.minimum(xmax, width)
        heatmap_ymax = tf.math.minimum(ymax, height)

        # finally, insert this patch into the heatmap
        indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)

        count = 0
        for j in tf.range(patch_ymin, patch_ymax):
            for i in tf.range(patch_xmin, patch_xmax):
                indices = indices.write(count,
                                        [heatmap_ymin + j, heatmap_xmin + i])
                updates = updates.write(count, gaussian_patch[j][i])
                count += 1
        heatmap = tf.tensor_scatter_nd_update(heatmap, indices.stack(),
                                              updates.stack())

        # unfortunately, the code below doesn't work because
        # tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
        # heatmap[heatmap_ymin:heatmap_ymax, heatmap_xmin:heatmap_xmax] = gaussian_patch[patch_ymin:patch_ymax,patch_xmin:patch_xmax]

        return heatmap

    def make_heatmaps(self, features):
        v = tf.cast(
            tf.sparse.to_dense(features['image/object/parts/v']),
            dtype=tf.float32)
        x = tf.cast(
            tf.math.round(
                tf.sparse.to_dense(features['image/object/parts/x']) *
                self.heatmap_shape[0]),
            dtype=tf.int32)
        y = tf.cast(
            tf.math.round(
                tf.sparse.to_dense(features['image/object/parts/y']) *
                self.heatmap_shape[1]),
            dtype=tf.int32)

        num_heatmap = self.heatmap_shape[2]
        heatmap_array = tf.TensorArray(tf.float32, 16)

        for i in range(num_heatmap):
            if v[i] != 0:
                gaussian = self.generate_2d_guassian(
                    self.heatmap_shape[1], self.heatmap_shape[0], y[i], x[i])
                heatmap_array = heatmap_array.write(i, gaussian)
            else:
                heatmap_array = heatmap_array.write(
                    i, tf.zeros((self.heatmap_shape[1],
                                 self.heatmap_shape[0])))

        heatmaps = heatmap_array.stack()
        heatmaps = tf.transpose(
            heatmaps, perm=[1, 2, 0])  # change to (64, 64, 16)
        return heatmaps

    def parse_tfexample(self, example_proto):
        image_feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(example_proto,
                                          image_feature_description)