# Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)

ILSVRC2012 dataset is around 150G unpacked. Flatten dataset for PyTorch requires another 150G, and generate TFRecord for TensorFlow requires another 150G

Create a directory to store the dataset under `imagenet-2012` directory. 
```
mkdir -p ./dataset
cd ./dataset
```

Download it from here: [http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads)

```bash
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_train_v2.tar.gz
wget http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_val_v3.tgz
```

Once you have them downloaded, untar them into these directories

```bash
mkdir ./train
tar xvf ILSVRC2012_img_train.tar -C ./train
mkdir ./val
tar xvf ILSVRC2012_img_val.tar -C ./val
mkdir ./test
tar xvf ILSVRC2012_img_test.tar -C ./test
mkdir ./bbox_train
tar xvf ILSVRC2012_bbox_train_v2.tar.gz -C ./bbox_train
mkdir ./bbox_val
tar xvf ILSVRC2012_bbox_val_v3.tgz -C ./bbox_val
```

After downloading and unpacking all data, now you need to preprocess them for different framework

## TensorFlow
Inside `../Datasets` directory:

Generate bbox csv from training bbox xml
```bash
sudo python3 process_bounding_boxes.py ../dataset/bbox_train/ ./imagenet_2012_synsets.txt > imagenet_2012_bounding_boxes.csv
```
Create a `tfrecord` directory in your `dataset` directory
```bash
mkdir -p ../dataset/tfrecord
```
Build TFRecord using the following script and command. You might need to tweak num_threads based on your machine CPUs to achieve best performance.
```bash
sudo nohup python3 build_imagenet_tfrecord.py --output_directory ../dataset/tfrecord/ --num_threads 16 --train_directory ../dataset/train --validation_directory ../dataset/val &
```
Wait for a while, then you will have:
```
tfrecord/train-00000-of-01024
tfrecord/train-00001-of-01024
...
tfrecord/train-01023-of-01024

tfrecord/validation-00000-of-00128
tfrecord/validation-00001-of-00128
...
tfrecord/validation-00127-of-00128
```
finally, move them to their own folder:
```
sudo mkdir -p tfrecord_train && sudo mv train* tfrecord_train
sudo mkdir -p tfrecord_val && sudo mv validation* tfrecord_val
```
## PyTorch

You might need `sudo` if you have permission issue, or `nohup` if you need to walk away while waiting

After decompressing all files, you should have `train` folder filled with things like `n03249569.tar`. Copy `Datasets/ILSVRC2012/untar-script` to your `train` folder, and run:
```
./untar-script.sh
```

Then you will have those `n03249569.tar` all decompress into their own folder like `n03249569`. Remove tar files:
```
rm -rf ./*.tar
```
Next, you will need to further flatten the directories since the data loader is designed to read from one directory. Copy `Datasets/ILSVRC2012/flatten-script` to your root `dataset` folder, and run:
```
./flatten-script.sh
```

This would take a while, and move all 1.28M images into `train_flatten` directory

Then go to your `val` folder, run [this script](https://github.com/juliensimon/aws/blob/master/mxnet/imagenet/build_validation_tree.sh) to group them by annotation first. And then copy `./Datasets/ILSVRC2012/flatten-val-script.sh` to `/val` folder, run it:
```
./flatten-val-script.sh
mv ./val-flatten ../
```
Finally, download [this](https://github.com/juliensimon/aws/blob/master/mxnet/imagenet/synsets_with_descriptions.txt) so that you know how to map id to real annotation name and save it as `synsets.txt` in your `dataset` directory

Finally, your dataset directory should look like this:
```
imagenet-2012
    |_dataset
        |_train
        |   |_n04347754
        |   |_...
        |_train_flatten
        |   |_n01440764_10026.JPEG
        |   |_n01440764_10027.JPEG
        |   |_...
        |_val
        |   |_n04347754
        |   |_...
        |_val_flatten
        |   |_n04548280_ILSVRC2012_val_00030987.JPEG
        |   |_n04548280_ILSVRC2012_val_00030330.JPEG
        |   |_...
        |_test
        |_bbox_train
        |_bbox_val
        |_synsets.txt
```