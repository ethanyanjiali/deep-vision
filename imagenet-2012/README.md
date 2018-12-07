# ImageNet ILSVRC2012

## Set Up Dataset

Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)

Create a directory to store the dataset. 
```
mkdir -p ./dataset
cd ./dataset
```

Download it from here: [http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads)

Once you have `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`:
```bash
mkdir ./train
tar xvf ILSVRC2012_img_train.tar -C ./train
mkdir ./val
tar xvf ILSVRC2012_img_val.tar -C ./val
```

You might need `sudo` if you have permission issue, or `nohup` if you need to walk away while waiting

After decompressing all files, you should have `train` folder filled with things like `n03249569.tar`. Copy `untar-script` to your `train` folder, and run:
```
./untar-script.sh
```

Then you will have those `n03249569.tar` all decompress into their own folder like `n03249569`. Remove tar files:
```
rm -rf ./*.tar
```
Next, you will need to further flatten the directories since the data loader is designed to read from one directory. Copy `flatten-script` to your root `dataset` folder, and run:
```
./flatten-script.sh
```

This would take a while, and move all 1.28M images into `train_flatten` directory

Then go to your `val` folder, run [this script](https://github.com/juliensimon/aws/blob/master/mxnet/imagenet/build_validation_tree.sh) to group them by annotation first. And then copy `./script/flatten-val-script.sh` to `/val` folder, run it:
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
        |_synsets.txt
```

## Start Training

This repo implements many different models. Once you have the dataset ready, you can start the training code by running one of the commands below in this directory:

| Model                                                                                                                             | Command               |
|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| [AlexNet Original Version](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) | `make train_alexnet1` |
| [AlexNet Second Version](https://arxiv.org/pdf/1404.5997.pdf)                                                                     | `make train_alexnet2` |
|                                                                                                                                   |                       |