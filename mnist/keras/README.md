# MNIST - Keras 

This directory contains the Keras implementation of famous classification models using public MNIST dataset

## Directory Structure

```
|__notebooks        // Jupyter notebooks to visualize the model inference
|__logs             // training logs for the models I trained
|__models           // model network structure implementations
|__tensorboard_logs // TensorBoard event files
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
- To run the notebook, please download the pretrained model and loggers file to `saved_model` directory first.

## LeNet

Training Command for LeNet-5:
```
make train_lenet5
```
There're multiple variation of LeNet, and here I've only implemented LeNet-5 as an example. Unlike mordern deep learning network architecture (as of 2018), LeNet was introduced in 1998, therefore some caveats needs to be mentioned:

- I didn't follow the subset of feature map that shows in Table I of the paper. That was used to reduce computational requirement but we should be fine nowadays.

**Test Accuracy**: 98.76% (Top-1)

**Training Log**: [lenet5-keras-yanjiali-010219.log](logs/lenet5-keras-yanjiali-010219.log)

**Pretrained Model File**: [lenet5-keras-yanjiali-010219.hdf5](https://drive.google.com/file/d/1fuEj-mKFNltFYHn4HBNgJAH9eaiVgvWM/view?usp=sharing)

**Notebook Visualization**: [LeNet.ipynb](notebooks/LeNet.ipynb), loggers file [lenet5-keras-loggers-010519.pkl](https://drive.google.com/file/d/1YcfCy1B3SO6nO7BiUkU2770SKzorNUt9/view?usp=sharing)