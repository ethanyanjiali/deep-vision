# ImageNet ILSVRC2012 - TensorFlow / Keras 

This directory contains the TensorFlow implementation (Keras APIs) to replicate famous classification models with public ImageNet ILSVRC2012 dataset

## Directory Structure

```
tensorflow
|__models           // model network structure pytorch implementations
|__data_load.py     // data preprocessing script
|__train.py         // single machine training script
```

## Set Up Dataset

Follow the instruction here [DATASET.md](../DATASET.md) to download the ILSVRC2012 dataset first.

## AlexNet
### AlexNet V2
```
make train_alexnet2
```
[Source Code](models/alexnet_v2.py)

## ResNet

### ResNet-50
```
make train_resnet50
```
[Source Code](models/resnet50.py)

### ResNet-152
```
make train_resnet152
```
[Source Code](models/resnet152.py)