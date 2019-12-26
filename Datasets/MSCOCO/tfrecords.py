import csv
import io
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
import ray
import tensorflow as tf

num_train_shards = 64
num_val_shards = 8
ray.init()
tf.get_logger().setLevel('ERROR')


def chunkify(l, n):
    size = len(l) // n
    start = 0
    results = []
    for i in range(n - 1):
        results.append(l[start:start + size])
        start += size
    results.append(l[start:])
    return results


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def genreate_tfexample(anno_list):
    filename = anno_list[0]['filename']
    with open(filename, 'rb') as image_file:
        content = image_file.read()

    image = Image.open(filename)
    if image.format != 'JPEG' or image.mode != 'RGB':
        image_rgb = image.convert('RGB')
        with io.BytesIO() as output:
            image_rgb.save(output, format="JPEG", quality=95)
            content = output.getvalue()

    width, height = image.size
    depth = 3

    class_ids = []
    class_texts = []
    bbox_xmins = []
    bbox_ymins = []
    bbox_xmaxs = []
    bbox_ymaxs = []
    for anno in anno_list:
        class_ids.append(anno['class_id'])
        class_texts.append(anno['class_text'].encode())
        xmin, ymin, xmax, ymax = anno['xmin'], anno['ymin'], anno[
            'xmax'], anno['ymax']
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = float(
            xmin) / width, float(ymin) / height, float(xmax) / width, float(
                ymax) / height
        assert bbox_xmin <= 1 and bbox_xmin >= 0
        assert bbox_ymin <= 1 and bbox_ymin >= 0
        assert bbox_xmax <= 1 and bbox_xmax >= 0
        assert bbox_ymax <= 1 and bbox_ymax >= 0
        bbox_xmins.append(bbox_xmin)
        bbox_ymins.append(bbox_ymin)
        bbox_xmaxs.append(bbox_xmax)
        bbox_ymaxs.append(bbox_ymax)

    feature = {
        'image/height':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/depth':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
        'image/object/bbox/xmin':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_xmins)),
        'image/object/bbox/ymin':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_ymins)),
        'image/object/bbox/xmax':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_xmaxs)),
        'image/object/bbox/ymax':
        tf.train.Feature(float_list=tf.train.FloatList(value=bbox_ymaxs)),
        'image/object/class/label':
        tf.train.Feature(int64_list=tf.train.Int64List(value=class_ids)),
        'image/object/class/text':
        tf.train.Feature(bytes_list=tf.train.BytesList(value=class_texts)),
        'image/encoded':
        _bytes_feature(content),
        'image/filename':
        _bytes_feature(os.path.basename(filename).encode())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


@ray.remote
def build_single_tfrecord(chunk, path):
    print('start to build tf records for ' + path)

    with tf.io.TFRecordWriter(path) as writer:
        for anno_list in chunk:
            tf_example = genreate_tfexample(anno_list)
            writer.write(tf_example.SerializeToString())

    print('finished building tf records for ' + path)


def build_tf_records(annotations, total_shards, split):
    annotations_by_image = {}
    for annotation in annotations:
        if annotation['filename'] in annotations_by_image:
            annotations_by_image[annotation['filename']].append(annotation)
        else:
            annotations_by_image[annotation['filename']] = [annotation]
    chunks = chunkify(list(annotations_by_image.values()), total_shards)
    futures = [
        # train_0001_of_0064.tfrecords
        build_single_tfrecord.remote(
            chunk, './tfrecords/{}_{}_of_{}.tfrecords'.format(
                split,
                str(i + 1).zfill(4),
                str(total_shards).zfill(4),
            )) for i, chunk in enumerate(chunks)
    ]
    ray.get(futures)


def parse_one_annotation(anno, categories, dir):
    category_id = int(anno['category_id'])
    category = categories[category_id]
    # ms coco class id starts from 1 because 0 is reserved for background
    # but i want it to start from 0
    class_id = category[0]
    if class_id < 0:
        print('ALERT: class is {} is invalid'.format(class_id))
    class_text = category[1]
    bbox = anno['bbox']
    filename = '{}/{}.jpg'.format(dir, str(anno['image_id']).rjust(12, '0'))
    annotation = {
        'filename': filename,
        'class_id': class_id,
        'class_text': class_text,
        # http://cocodataset.org/#format-data
        # bbox is in format of (x, y, width, height)
        'xmin': float(bbox[0]),
        'ymin': float(bbox[1]),
        'xmax': float(bbox[0]) + float(bbox[2]),
        'ymax': float(bbox[1]) + float(bbox[3]),
    }
    return annotation


def main():
    print('Start to parse annotations.')
    if not os.path.exists('./tfrecords'):
        os.makedirs('./tfrecords')

    with open('./annotations/instances_train2017.json') as train_json:
        train_annos = json.load(train_json)
        train_categories = {
            category['id']: (i, category['name'])
            for i, category in enumerate(train_annos['categories'])
        }
        print(train_categories)
        train_annotations = [
            parse_one_annotation(anno, train_categories, './train2017')
            for anno in train_annos['annotations']
        ]
        del (train_annos)

    with open('./annotations/instances_val2017.json') as val_json:
        val_annos = json.load(val_json)
        val_categories = {
            category['id']: (i, category['name'])
            for i, category in enumerate(val_annos['categories'])
        }
        print(val_categories)
        val_annotations = [
            parse_one_annotation(anno, val_categories, './val2017')
            for anno in val_annos['annotations']
        ]
        del (val_annos)

    print('Start to build TF Records.')
    build_tf_records(train_annotations, num_train_shards, 'train')
    build_tf_records(val_annotations, num_val_shards, 'val')

    print('Successfully wrote {} annotations to TF Records.'.format(
        len(train_annotations) + len(val_annotations)))


if __name__ == '__main__':
    main()
