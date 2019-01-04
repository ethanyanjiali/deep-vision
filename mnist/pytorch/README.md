# MNIST - PyTorch

This directory contains the PyTorch implementation of famous classification models using public MNIST dataset

## Directory Structure

```
|__notebooks        // Jupyter notebooks to visualize the model inference
|__logs             // training logs for the models I trained
|__models           // model network structure implementations
|__data_load.py     // data loader
|__train.py         // single machine training script
```

## System Requirement

MNIST dataset doesn't require GPU and high-end hardware like other dataset. You should be able to converge within reasonable amount of time by using your own machine CPU.

## Start Training

This repo implements some popular models that uses MNIST dataset. Once you have the dataset ready, you can start the training code by running tasks in the Makefile. I trained some of the implemented models, and provided the training log and model file for your reference.

There're few tips before you acutally start training:

- I use Python 3 for this project.
- Make sure you have set up virtual environment and also installed dependencies by `pip install -r requirements.in`.
- There're multiple options defined in `train.py`. For example, model to train `-m`.
- There're also some examples for how to resume previous paused training in the Makefile.
- To run the notebook, please download the pretrained model to `saved_model` directory first.

## LeNet

Training Command for LeNet-5:
```
make train_lenet5
```
There're multiple variation of LeNet, and here I've only implemented LeNet-5 as an example. It acheives 98.72% accuracy on validation set. Unlike mordern deep learning network architecture (as of 2018), LeNet was introduced in 1998, therefore some caveats needs to be mentioned:

- Part of the implementation is different from the original [paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf). I tried my best to mimic the original design but some old techniques are not supported in PyTorch which is a relative new framework. Also it helps to avoid some hassle.
- One of the difference is that I didn't follow the subset of feature map that shows in Table I of the paper. That was used to reduce computational requirement but we should be fine nowadays.
- The other difference is that instead of using RBF, I used a 10-way softmax as output. This makes things easier.
- Initially I used val_loss as metrics for LR scheduler, but changed to val_top1_acc later. The training log reflects the former.

**Training Log**: [lenet5-pt-yanjiali-010219.log](logs/lenet5-pt-yanjiali-010219.log)

**Pretrained Model File**: [lenet5-pt-yanjiali-010219.pt](https://drive.google.com/file/d/1lrvO1aRgE9aMSTJJbb3Gx4wriu7PcdOu/view?usp=sharing)

**Notebook Visualization**: [lenet.ipynb](notebooks/lenet.ipynb)