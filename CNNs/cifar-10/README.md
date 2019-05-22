# CIFAR-10

The [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) is labeled subset of the 80 million tiny images dataset. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

## Set Up Dataset

Before you start training, you need to download CIFAR-10 dataset from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html), find `CIFAR-10 python version` or click [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) to download the `tar.gz` file. Then create a `dataset` directory under `cifar-10` and untar that file into `dataset` directory.

Finally, your `dataset` directory should look like this:
```
dataset
|__file.txt~
|__meta
|__test
|__train
```
