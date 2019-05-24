import os
import glob
import argparse

import tensorflow as tf
from PIL import Image


def convert_to_tfexample(img_path):
    try:
        with open(img_path, 'rb') as f:
            content = f.read()
        with Image.open(img_path) as im:
            im.load()
            if im.format != 'JPEG':
                print('Wrong image format, path {}, format {}'.format(img_path, im.format))
            assert (im.format == 'JPEG')
            filename = os.path.basename(img_path)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content])),
                        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['JPEG'.encode()])),
                        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[im.width])),
                        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[im.height])),
                        'image/filename': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[filename.encode()]))
                    }))
            return example
    except Exception as e:
        print(e)
        return None


def main():
    parser = argparse.ArgumentParser(description='Convert iFood 2018 dataset to TFRecord files.')
    parser.add_argument(
        '--dataset', help='The name of the dataset')
    args = parser.parse_args()

    train_A_files = sorted(glob.glob(os.path.join('datasets', args.dataset, 'trainA', '*')))
    train_B_files = sorted(glob.glob(os.path.join('datasets', args.dataset, 'trainB', '*')))
    test_A_files = sorted(glob.glob(os.path.join('datasets', args.dataset, 'testA', '*')))
    test_B_files = sorted(glob.glob(os.path.join('datasets', args.dataset, 'testB', '*')))

    train_A_output = os.path.join('tfrecords', args.dataset, 'trainA.tfrecord')
    train_B_output = os.path.join('tfrecords', args.dataset, 'trainB.tfrecord')
    test_A_output = os.path.join('tfrecords', args.dataset, 'testA.tfrecord')
    test_B_output = os.path.join('tfrecords', args.dataset, 'testB.tfrecord')

    with tf.io.TFRecordWriter(train_A_output) as writer:
        for file in train_A_files:
            example = convert_to_tfexample(file)
            writer.write(example.SerializeToString())
    print('Finished converting TFRecords for trainA')
    with tf.io.TFRecordWriter(train_B_output) as writer:
        for file in train_B_files:
            example = convert_to_tfexample(file)
            writer.write(example.SerializeToString())
    print('Finished converting TFRecords for trainB')
    with tf.io.TFRecordWriter(test_A_output) as writer:
        for file in test_A_files:
            example = convert_to_tfexample(file)
            writer.write(example.SerializeToString())
    print('Finished converting TFRecords for testA')
    with tf.io.TFRecordWriter(test_B_output) as writer:
        for file in test_B_files:
            example = convert_to_tfexample(file)
            writer.write(example.SerializeToString())
    print('Finished converting TFRecords for testB')


if __name__ == '__main__':
    main()