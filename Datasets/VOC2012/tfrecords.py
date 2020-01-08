from xml.etree import ElementTree as ET
import io
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
import ray
import tensorflow as tf

num_train_shards = 4
num_val_shards = 4
num_test_shards = 4
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


def genreate_tfexample(anno):
    with open(anno['filepath'], 'rb') as image_file:
        content = image_file.read()
    width = anno.get('width', -1)
    height = anno.get('height', -1)
    depth = anno.get('depth', -1)
    if depth != 3 and depth != -1:
        print('WANRNING: Image {} has depth of {}'.format(
            anno['filename'], depth))
    class_ids = []
    class_texts = []
    bbox_xmins = []
    bbox_ymins = []
    bbox_xmaxs = []
    bbox_ymaxs = []
    for bbox in anno.get('bboxes', []):
        class_ids.append(bbox['class_id'])
        class_texts.append(bbox['class_text'].encode())
        xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox[
            'xmax'], bbox['ymax']
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
        _bytes_feature(anno['filename'].encode())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


@ray.remote
def build_single_tfrecord(chunk, path):
    print('start to build tf records for ' + path)

    with tf.io.TFRecordWriter(path) as writer:
        for anno in chunk:
            tf_example = genreate_tfexample(anno)
            writer.write(tf_example.SerializeToString())

    print('finished building tf records for ' + path)


def build_tf_records(annotations, total_shards, split):
    chunks = chunkify(annotations, total_shards)
    futures = [
        # train_0001_of_0064.tfrecords
        build_single_tfrecord.remote(
            chunk, './tfrecords_voc_2012/{}_{}_of_{}.tfrecords'.format(
                split,
                str(i + 1).zfill(4),
                str(total_shards).zfill(4),
            )) for i, chunk in enumerate(chunks)
    ]
    ray.get(futures)


def parse_one_xml(xml_file, names_map):
    tree = ET.parse(os.path.join('./VOCdevkit/VOC2012/Annotations', xml_file))
    root = tree.getroot()
    filename = root.find('.//filename').text
    filepath = os.path.join('./VOCdevkit/VOC2012/JPEGImages', filename)
    objects_els = root.findall('.//object')
    size_el = root.find('size')
    width = int(size_el.find('width').text)
    height = int(size_el.find('height').text)
    depth = int(size_el.find('depth').text)

    bboxes = []
    for obj_el in objects_els:
        name_el = obj_el.find('name')
        bbox_el = obj_el.find('bndbox')
        bboxes.append({
            'class_text': name_el.text,
            'class_id': names_map[name_el.text],
            'xmin': int(bbox_el.find('xmin').text),
            'ymin': int(bbox_el.find('ymin').text),
            'xmax': int(bbox_el.find('xmax').text),
            'ymax': int(bbox_el.find('ymax').text),
        })

    return {
        'filepath': filepath,
        'filename': filename,
        'width': width,
        'height': height,
        'depth': depth,
        'bboxes': bboxes,
    }


def main():
    print('Start to parse annotations.')
    if not os.path.exists('./tfrecords_voc_2012'):
        os.makedirs('./tfrecords_voc_2012')

    train_val_split = {}
    with open('./VOCdevkit/VOC2012/ImageSets/Main/train.txt') as train_fp:
        lines = train_fp.read().splitlines()
        for line in lines:
            train_val_split[line] = 'train'
    with open('./VOCdevkit/VOC2012/ImageSets/Main/val.txt') as val_fp:
        lines = val_fp.read().splitlines()
        for line in lines:
            train_val_split[line] = 'val'
    with open('./VOCdevkit/VOC2012/ImageSets/Main/test.txt') as test_fp:
        lines = test_fp.read().splitlines()
        for line in lines:
            train_val_split[line] = 'test'

    with open('./voc_2012_names.txt') as names_fp:
        names = names_fp.read().splitlines()
        names_map = {name: i for i, name in enumerate(names)}
    print(names_map)

    train_annotations = []
    val_annotations = []
    test_annotations = []
    for xml_file in os.listdir('./VOCdevkit/VOC2012/Annotations'):
        image_id = xml_file[:-4]
        split = train_val_split.get(image_id)
        if split == 'train':
            train_annotations.append(parse_one_xml(xml_file, names_map))
        elif split == 'val':
            val_annotations.append(parse_one_xml(xml_file, names_map))
        else:
            filename = image_id + '.jpg'
            filepath = os.path.join('./VOCdevkit/VOC2012/JPEGImages', filename)
            test_annotations.append({
                'filepath': filepath,
                'filename': filename,
            })

    print('Start to build TF Records.')
    build_tf_records(train_annotations, num_train_shards, 'train')
    build_tf_records(val_annotations, num_val_shards, 'val')
    build_tf_records(test_annotations, num_test_shards, 'test')

    print('Successfully wrote {} annotations to TF Records.'.format(
        len(train_annotations) + len(val_annotations) + len(test_annotations)))


if __name__ == '__main__':
    main()
