# ResNet - ImageNet ILSVRC2012 - TensorFlow / Keras 

This directory contains the TensorFlow implementation (Keras APIs) to replicate ResNet with public ImageNet ILSVRC2012 dataset

## Directory Structure

```
tensorflow
|__models           // model network structure pytorch implementations
|__data_load.py     // data preprocessing script
|__train.py         // single machine training script
```

## Set Up Dataset

Follow the instruction here [DATASET.md](../../Datasets/ILSVRC2012/DATASET.md) to download the ILSVRC2012 dataset first.

## ResNet

### ResNet-50 V1
```
make train_resnet50
```
[Source Code](models/resnet50.py)

### ResNet-152 V1
```
make train_resnet152
```
[Source Code](models/resnet152.py)

### ResNet-50 V2

I haven't tested ResNetV2 in this `train.py` script yet. But it should work.

[Source Code](models/resnet50v2.py)