# MNIST - PyTorch

This directory contains the PyTorch implementation of LeNet using public MNIST dataset

## Directory Structure

```
|__notebooks        // Jupyter notebooks to visualize the model inference
|__logs             // training logs for the models I trained
|__models           // model network structure implementations
|__data_load.py     // data loader
|__train.py         // single machine training script
```

## Set up Dataset

Follow this [DATASET.md](../../Datasets/MNIST/DATASET.md)

## System Requirement

MNIST dataset doesn't require GPU and high-end hardware like other dataset. You should be able to converge within reasonable amount of time by using your own machine CPU.

## Start Training

This repo implements some popular models that uses MNIST dataset. Once you have the dataset ready, you can start the training code by running tasks in the Makefile. I trained some of the implemented models, and provided the training log and model file for your reference.

There're few tips before you acutally start training:

- There're multiple options defined in `train.py`. For example, model to train `-m`.
- There're also some examples for how to resume previous paused training in the Makefile.
- To run the notebook, please download the pretrained model to `saved_model` directory first.

## LeNet

Training Command for LeNet-5:
```
make train_lenet5
```
[Source Code](models/lenet5.py)

There're multiple variation of LeNet, and here I've only implemented LeNet-5 as an example. Unlike mordern deep learning network architecture (as of 2018), LeNet was introduced in 1998, therefore some caveats needs to be mentioned:

- Part of the implementation is different from the original [paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf). I tried my best to mimic the original design but some old techniques are not supported in PyTorch which is a relative new framework. Also it helps to avoid some hassle.
- One of the difference is that I didn't follow the subset of feature map that shows in Table I of the paper. That was used to reduce computational requirement but we should be fine nowadays.
- The other difference is that instead of using RBF, I used a 10-way softmax as output. This makes things easier.

**Test Accuracy**: 99.07% (Top-1)

**Training Log**: [lenet5-pt-yanjiali-010619.log](logs/lenet5-pt-yanjiali-010619.log)

**Pretrained Model File**: [lenet5-pt-yanjiali-010619.pt](https://drive.google.com/file/d/1fMD2i3bIEK-fU9JMiBGQ5EVv-A7dFJux/view?usp=sharing)

**Notebook Visualization**: [LeNet.ipynb](notebooks/LeNet.ipynb)