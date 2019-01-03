# ImageNet ILSVRC2012 - PyTorch

This directory contains the PyTorch code to replicate famous cliassification models with public ImageNet ILSVRC2012 dataset

## Directory Structure

```
pytorch
|__notebooks        // Jupyter notebooks to visualize the model inference
|__logs             // training logs for the models I trained
|__test_images      // test images for inference visuliazation
|__models           // model network structure pytorch implementations
|__data_load.py     // data loader
|__train.py         // single machine training script
|__train_dist.py    // distributed training script
```

## Set Up Dataset

Follow the instruction here [DATASET.md](../DATASET.md) to download the ILSVRC2012 dataset first.

## System Requirement

I'm training all these models with 8 vCPU, 24GB RAM and one Nvidia P100 GPU (~16G). If you have different hardware, you might need to change some parameters to fit model with your hardware, especially batch size (when memory exceed) and num_workers (for multi-thread loading).

## Start Training

This repo implements many different models. Once you have the dataset ready, you can start the training code by running tasks in the Makefile. I trained some of the implemented models, and provided the training log and model file for your reference.

There're few tips before you acutally start training:

- I use Python 3 for this project
- Make sure you have set up virtual environment and also installed dependencies by `pip install -r requirements.in`
- There're multiple options defined in `train.py`. For example, model to train `-m`, and checkpoint file to use `-c`.
- There're also some examples for how to resume previous paused training in the Makefile.
- To run the notebook, please download the pretrained model to `saved_model` directory first.

## AlexNet
Training Command for V1 and V2:

### AlexNet V1
```
make train_alexnet1
```

### AlexNet V2
```
make train_alexnet2
```
Among two versions, I trained AlexNet V2 which achieves. Some training notes: 

- I manually changed learning rate multiple times through training. But used a LR scheduler to in the final version with milestone of 30, 45, 80, 95 epochs and 0.1 decay.
- With default weight init from PyTorch, the loss will stop decreasing around 4.5 so I used Kaiming init instead.
- Data augmentation part is not exactly same with the original paper.
- I modified the log format few times during training, so the log file is not consistent everywhere. 

**Accuracy**: 50.98% (Top-1)

**Training Log**: [alexnet2.txt](logs/alexnet2.txt)

**Pretrained Model File**: [alexnet_v2_yanjiali_12_26_18.pt](https://drive.google.com/file/d/1EGYtcLsEV2ZFv6sSCKNd_fSFxtwI2cYV/view?usp=sharing)

**Notebook Visualization**: [AlexNetV2.ipynb](notebooks/AlexNetV2.ipynb)

## VGG

## ResNet

### ResNet-34
```
make train_resnet34
```

- This model is trained by an old training script (`train_old.py`), therefore some log format might differ from those newer ones.
- Unlike the new training config, I trained resnet-34 with a batch size of 512. Since I used 8 GPUs, it's 64 batch size per GPU. Kaiming did mentioned that different batch size might degrad the accuracy. So I advice still stick to batch size of 256 when you use 8 GPUs.
- The best accuracy is achieved at epoch 93, but I trained until epoch 129

**Accuracy**: 68.96% (Top-1), 88.61% (Top-5)

**Training Log**: [resnet34-yanjiali-010319.log](logs/resnet34-yanjiali-010319.log)

**Pretrained Model File**: [resnet34-pt-yanjiali-010319.pt](https://drive.google.com/file/d/1M_LY94x1YYx5EYtqzQrXoASnUa-VcRHx/view?usp=sharing)

**Notebook Visualization**: [ResNet34.ipynb](notebooks/ResNet34.ipynb)

### ResNet-50

## FAQ

> I received `Missing key(s) in state_dict:...` when I load the PyTorch model file

It might because I used DataParallel to train. I tried to remove that DataParallel prefix from state dict, but in case you still see that, either load state dict with DataParallel wrapped model, or do the following:
```python
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v
```
For more infor: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686