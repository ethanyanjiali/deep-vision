import os
import time

import click
from google.cloud import storage

from train import train
"""
python3 main.py \
    --epochs=100 \
    --start_epoch=1 \
    --learning_rate=0.001 \
    --tensorboard_dir="./logs/" \
    --num_heatmap=16 \
    --batch_size=16 \
    --train_tfrecords="./dataset/tfrecords_mpii/train*" \
    --val_tfrecords="./dataset/tfrecords_mpii/val*"    
"""


@click.command()
@click.option('--epochs', default=100, help='Total number of epochs.')
@click.option(
    '--start_epoch', default=1, help='The initial epoch to start with.')
@click.option(
    '--learning_rate', default=0.001, help='The learning rate to start with.')
@click.option(
    '--tensorboard_dir',
    default="./logs",
    help='The directory to store Tensorboard events.')
@click.option('--checkpoint', help='The path to checkpoint file.')
@click.option('--num_heatmap', default=442, help='Number of heatmap layers.')
@click.option('--batch_size', default=32, help='Size of a mini batch.')
@click.option('--train_tfrecords', help='GCS location of training TF Records.')
@click.option('--val_tfrecords', help='GCS location of validation TF Records.')
@click.option(
    '--output_bucket', help='Bucket name of the model output GCS location.')
@click.option(
    '--output_dir', help='Directory name of the model output GCS location.')
@click.option('--version', help='Version number of the new model.')
def main(epochs, start_epoch, learning_rate, tensorboard_dir, checkpoint,
         num_heatmap, batch_size, train_tfrecords, val_tfrecords,
         output_bucket, output_dir, version):
    versioned_tensorboard_dir = os.path.join(tensorboard_dir, "v" + version)
    model_path = train(epochs, start_epoch, learning_rate,
                       versioned_tensorboard_dir, checkpoint, num_heatmap,
                       batch_size, train_tfrecords, val_tfrecords, version)
    print("Received model " + model_path)

    if output_bucket is None or output_dir is None:
        return

    storage_client = storage.Client()
    bucket = storage_client.bucket(output_bucket)

    file_name = os.path.basename(model_path)
    blob_name = os.path.join(output_dir, file_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(model_path)
    output = 'gs://' + os.path.join(output_bucket, blob_name)
    print("Uploaded model file to " + output)

    with open('/tmp/output.txt', 'w') as fp:
        fp.write(output + '\n')
        print("Saved output to /tmp/output.txt")


if __name__ == "__main__":
    main()