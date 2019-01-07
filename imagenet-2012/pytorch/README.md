# ImageNet ILSVRC2012 - PyTorch

This directory contains the PyTorch code to replicate famous classification models with public ImageNet ILSVRC2012 dataset

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

- There're multiple options defined in `train.py`. For example, model to train `-m`, and checkpoint file to use `-c`.
- There're also some examples for how to resume previous paused training in the Makefile.
- To run the notebook, please download the pretrained model to `saved_model` directory first.
- `data_load.py` implements some common data preprocessing and augmentation by using numpy. I could have use PyTorch built-in utils but this makes the process more clear

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

- The model could be further trained with smaller learning rate, but I stopped early to save computation resource
- I modified the data_load.py during training to fix that corrupted EXIF issue. So the second half of the training log doesn't have that warning
- Color jittering was not applied for this training

**Val Accuracy**: 57.69% (Top-1), 79.10% (Top-5)

**Training Log**: [alexnet2-yanjiali-010319.log](logs/alexnet2-yanjiali-010319.log)

**Pretrained Model File**: [alexnet_v2_yanjiali_12_26_18.pt](https://drive.google.com/file/d/1_leXoq7fAisfrK_ChZW5ziOzuO0kbb8N/view?usp=sharing)

**Notebook Visualization**: [AlexNetV2.ipynb](notebooks/AlexNetV2.ipynb)

## VGG

## Inception

### Inception V1 (GoogLeNet)
```bash
make train_inception1
```

- This GoogLeNet is trained by an old training script (`train_old.py`), therefore some log format might differ from those newer ones.
- From the notebook loss visualization, it seems like if I set max_epochs for lr policy to bigger number, maybe 90, could improve the performance when converge
- The best accuracy is achieved at epoch 83, but I trained until epoch 90
- Color jittering was not applied for this training

**Val Accuracy**: 69.58% (Top-1), 89.21% (Top-5)

**Training Log**: [inception1-yanjiali-010519.log](logs/inception1-yanjiali-010519.log)

**Pretrained Model File**: [inception1-pt-yanjiali-010519.pt](https://drive.google.com/file/d/1WdIUxW2nugfhLRUXE2xGg-ZvoZVVBfaF/view?usp=sharing)

**Notebook Visualization**: [GoogLeNet.ipynb](notebooks/GoogLeNet.ipynb)

### Inception V3

## ResNet

### ResNet-34
```
make train_resnet34
```

- This model is trained by an old training script (`train_old.py`), therefore some log format might differ from those newer ones.
- Unlike the new training config, I trained resnet-34 with a batch size of 512. Since I used 8 GPUs, it's 64 batch size per GPU. Kaiming did mentioned that different batch size might degrad the accuracy. So I advice still stick to batch size of 256 when you use 8 GPUs.
- The best accuracy is achieved at epoch 93, but I trained until epoch 129
- Color jittering was not applied for this training

**Val Accuracy**: 68.96% (Top-1), 88.61% (Top-5)

**Training Log**: [resnet34-yanjiali-010319.log](logs/resnet34-yanjiali-010319.log)

**Pretrained Model File**: [resnet34-pt-yanjiali-010319.pt](https://drive.google.com/file/d/1M_LY94x1YYx5EYtqzQrXoASnUa-VcRHx/view?usp=sharing)

**Notebook Visualization**: [ResNet34.ipynb](notebooks/ResNet34.ipynb)

### ResNet-50

## Load Pretrained Model

To load the model file I trained, please do:

```python
checkpoint = torch.load(
    '../saved_models/alexnet2-pt-yanjiali-010319.pt',
    map_location='cpu',
)
net.load_state_dict(checkpoint['model'])
```

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