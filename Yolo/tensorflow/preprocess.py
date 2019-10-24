import tensorflow as tf

num_classes = 601
global_batch_size = 32

def _decode_crop_and_flip(image_buffer, bbox, num_channels):
    """Crops the given image to a random part of the image, and randomly flips.

    We use the fused decode_and_crop op, which performs better than the two ops
    used separately in series, but note that this requires that the image be
    passed in as an un-decoded string Tensor.

    Args:
      image_buffer: scalar string Tensor representing the raw JPEG image buffer.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      num_channels: Integer depth of the image buffer for decoding.

    Returns:
      3-D tensor with cropped image.

    """
    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    shape = tf.image.extract_jpeg_shape(image_buffer)
    height = shape[0]
    width = shape[1]
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    normalized_window = tf.stack([offset_y/height, offset_x/width, (target_height+offset_y)/height, (target_width+offset_x)/width])

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped, normalized_window


def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    normalized_window = tf.stack([crop_top/height, crop_left/width, (crop_height+crop_top)/height, (crop_width+crop_left)/width])
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1]), normalized_window


def _mean_image_subtraction(image, means, num_channels):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
      num_channels: number of color channels in the image that will be distorted.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    return image - means


def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.

    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BILINEAR)


def preprocess_image(image_buffer,
                     bbox,
                     output_height,
                     output_width,
                     eval_resize_min,
                     num_channels=3,
                     is_training=False):
    """Preprocesses the given image.

    Preprocessing includes decoding, cropping, and resizing for both training
    and eval images. Training preprocessing, however, introduces some random
    distortion of the image to improve accuracy.

    Args:
      image_buffer: scalar string Tensor representing the raw JPEG image buffer.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      eval_resize_min: An integer indicating the size of the smallest side after
        resize. Only used in the evaluation cycle. For example, if an image is
        500 x 1000, it will be resized to eval_resize_min * (eval_resize_min * 2).
      num_channels: Integer depth of the image buffer for decoding. Defaults to
        3.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.

    Returns:
      A preprocessed image.
    """
    crop_window = None
    if is_training:
        # For training, we want to randomize some of the distortions.
        image, crop_window = _decode_crop_and_flip(image_buffer, bbox, num_channels)
        image = _resize_image(image, output_height, output_width)
    else:
        # For validation, we want to decode, resize, then just crop the middle.
        image = tf.io.decode_jpeg(image_buffer, channels=num_channels)
        image = _aspect_preserving_resize(image, eval_resize_min)
        image, crop_window = _central_crop(image, output_height, output_width)

    image.set_shape([output_height, output_width, num_channels])

#     return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels), crop_window
    return image, crop_window


def _parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields (values are included as examples):

      image/height: 462
      image/width: 581
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 615
      image/class/synset: 'n03623198'
      image/class/text: 'knee pad'
      image/object/bbox/xmin: 0.1
      image/object/bbox/xmax: 0.9
      image/object/bbox/ymin: 0.2
      image/object/bbox/ymax: 0.6
      image/object/bbox/label: 615
      image/format: 'JPEG'
      image/filename: 'ILSVRC2012_val_00041207.JPEG'
      image/encoded: <JPEG encoded string>

    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.

    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
        where each coordinate is [0, 1) and the coordinates are arranged as
        [ymin, xmin, ymax, xmax].
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature((), dtype=tf.string, default_value=''),
        'image/class/label': tf.io.VarLenFeature(tf.int64),
        'image/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update({
        k: sparse_float32 for k in
        ['image/object/bbox/xmin', 'image/object/bbox/ymin', 'image/object/bbox/xmax', 'image/object/bbox/ymax']
    })

    features = tf.io.parse_single_example(example_serialized, feature_map)
    label = tf.cast(tf.sparse.to_dense(sp_input=features['image/class/label']), tf.int32)
    bbox_label = tf.cast(tf.sparse.to_dense(sp_input=features['image/object/class/label']), tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, bbox_label


def preprocess_label(bbox, bbox_label, crop_window):
    crop_window = tf.cast(crop_window, tf.float32)
    crop_window_x1 = crop_window[0]
    crop_window_y1 = crop_window[1]
    crop_window_x2 = crop_window[2]
    crop_window_y2 = crop_window[3]
    
    def calculate_coverage_ratio(one_box):
        one_box = tf.cast(one_box, tf.float32)
        one_box_x1 = one_box[0]
        one_box_y1 = one_box[1]
        one_box_x2 = one_box[2]
        one_box_y2 = one_box[3]
    
        x_left = tf.maximum(crop_window_x1, one_box_x1)
        y_top = tf.maximum(crop_window_y1, one_box_y1)
        x_right = tf.minimum(crop_window_x2, one_box_x2)
        y_bottom = tf.minimum(crop_window_y2, one_box_y2)
        
        intersection_area = tf.maximum(x_right - x_left, tf.constant(0.0)) * tf.maximum(y_bottom - y_top, tf.constant(0.0))
        crop_window_area = (crop_window_x2 - crop_window_x1) * (crop_window_x2 - crop_window_y1)
        one_box_area = (one_box_x2 - one_box_x1) * (one_box_y2 - one_box_y1)
        
        coverage = intersection_area / one_box_area
        return tf.greater_equal(coverage, tf.constant(0.05))
        
    positive_mask = tf.map_fn(calculate_coverage_ratio, bbox[0], dtype=tf.bool)
    positive_label, _ = tf.unique(tf.boolean_mask(bbox_label, positive_mask))

    return positive_label


def parse_record(raw_record):
    """Parses a record containing a training example of an image.

    The input record is parsed into a label and image, and the image is passed
    through preprocessing steps (cropping, flipping, and so on).

    Args:
      raw_record: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
      is_training: A boolean denoting whether the input is for training.
      input_params: A dictionary containing relevant parameters required by the
        input_fn.

    Returns:
      Tuple with processed image tensor and multi-hot encoded label tensor.
    """
    output_size = 300
    image_buffer, label, bbox, bbox_label = _parse_example_proto(raw_record)
    
    image, crop_window = preprocess_image(
        image_buffer=image_buffer,
        bbox=bbox,
        output_height=output_size,
        output_width=output_size,
        num_channels=3,
        eval_resize_min=output_size,
        is_training=True)
    
    label = preprocess_label(
        bbox=bbox,
        bbox_label=bbox_label,
        crop_window=crop_window,
    )

    label = tf.cond(
        tf.equal(tf.size(label), 0), lambda: tf.zeros(601),
        lambda: tf.reduce_max(tf.one_hot(label, 601), axis=0))
    
    return image, label, bbox, bbox_label