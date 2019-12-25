# MNIST Dataset

The [MNIST](https://en.wikipedia.org/wiki/MNIST_database) database (Modified National Institute of Standards and Technology database) has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

## Set Up Dataset

(This step is not necessary for Keras)

The dataset has already been downloaded to `dataset` directory under `mnist`. The data loader will work when the dataset directory looks like this:
```
dataset
|__t10k-images-idx3-ubyte
|__t10k-labels-idx1-ubyte
|__train-images-idx3-ubyte
|__train-labels-idx3-ubyte
```
However, if you want to download your own copy, here's the link: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## Implementations

I've made the networks and training scripts in multiple frameworks. Please refer to their own directory for specific implementation.

- [PyTorch](pytorch)
    - LeNet-5
- [TensorFlow](tensorflow)
    - LeNet-5
    