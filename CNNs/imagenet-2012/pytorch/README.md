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
- There's an older version of training script called `train_old.py` which is used when I train some model. You should use `train.py` though because it's a refactored and improved version.
- There're also some examples for how to resume previous paused training in the Makefile.
- To run the notebook, please download the pretrained model to `saved_model` directory first.
- `data_load.py` implements some common data preprocessing and augmentation by using numpy. I could have use PyTorch built-in utils but this makes the process more clear

## AlexNet

### AlexNet V1
```
make train_alexnet1
```
[Source Code](models/alexnet_v1.py)

### AlexNet V2
```
make train_alexnet2
```
[Source Code](models/alexnet_v2.py)

Among two versions, I trained AlexNet V2 which achieves. Some training notes: 

- The model could be further trained with smaller learning rate, but I stopped early to save computation resource
- I modified the data_load.py during training to fix that corrupted EXIF issue. So the second half of the training log doesn't have that warning
- Color jittering was not applied for this training

**Val Accuracy**: 57.69% (Top-1), 79.10% (Top-5)

**Training Log**: [alexnet2-yanjiali-010319.log](logs/alexnet2-yanjiali-010319.log)

**Pretrained Model File**: [alexnet2-pt-yanjiali-010319.pt](https://drive.google.com/file/d/1_leXoq7fAisfrK_ChZW5ziOzuO0kbb8N/view?usp=sharing)

**Notebook Visualization**: [AlexNetV2.ipynb](notebooks/AlexNetV2.ipynb)

## VGG

### VGG-16
```bash
make train_vgg16
```
[Source Code](models/vgg16.py)

- I modified the data_load.py during training to fix that corrupted EXIF issue. So the second half of the training log doesn't have that warning. Also it used the old training script `train_old.py`.
- Color jittering was not applied for this training

**Val Accuracy**: 69.21% (Top-1), 88.67% (Top-5)

**Training Log**: [vgg16-yanjiali-011219.log](logs/vgg16-yanjiali-011219.log)

**Pretrained Model File**: [vgg16-pt-yanjiali-011219.pt](https://drive.google.com/file/d/1774N5eUqXhPvcb2NJg6e3412ddNesGEW/view?usp=sharing)

**Notebook Visualization**: [VGG16.ipynb](notebooks/VGG16.ipynb)

### VGG-19
```bash
make train_vgg19
```
[Source Code](models/vgg19.py)

- This model is trained with the new training script `train.py`
- I modified the batch size from 128 to 64 in the middle of the training because I got a CUDA memory error

**Val Accuracy**: 70.04% (Top-1), 89.30% (Top-5)

**Training Log**: [vgg19-yanjiali-011819.log](logs/vgg19-yanjiali-011819.log)

**Pretrained Model File**: [vgg19-pt-yanjiali-011819.pt](https://drive.google.com/file/d/1cIfy6QuF_b9BMdHORuCkjCsRz3nx2RHG/view?usp=sharing)

**Notebook Visualization**: [VGG19.ipynb](notebooks/VGG19.ipynb)

## Inception

### Inception V1 (GoogLeNet)
```bash
make train_inception1
```
[Source Code](models/inception_v1.py)

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
[Source Code](models/resnet34.py)

- This model is trained with an old training script (`train_old.py`), therefore some log format might differ from those newer ones.
- Unlike the new training config, I trained resnet-34 with a batch size of 512. Since I used 8 GPUs, it's 64 batch size per GPU. Kaiming did mentioned that different batch size might degrad the accuracy. So I advice still stick to batch size of 256 when you use 8 GPUs.
- The best accuracy is achieved at epoch 93, but I trained until epoch 129
- Color jittering was not applied for this training

**Val Accuracy**: 68.96% (Top-1), 88.61% (Top-5)

**Training Log**: [resnet34-yanjiali-010319.log](logs/resnet34-yanjiali-010319.log)

**Pretrained Model File**: [resnet34-pt-yanjiali-010319.pt](https://drive.google.com/file/d/1M_LY94x1YYx5EYtqzQrXoASnUa-VcRHx/view?usp=sharing)

**Notebook Visualization**: [ResNet34.ipynb](notebooks/ResNet34.ipynb)

### ResNet-50
```
make train_resnet50
```
[Source Code](models/resnet50.py)

- This model is trained with the new training script `train.py`

**Val Accuracy**: 73.93% (Top-1), 91.54% (Top-5)

**Training Log**: [resnet50-yanjiali-010919.log](logs/resnet50-yanjiali-010919.log)

**Pretrained Model File**: [resnet50-pt-yanjiali-010919.pt](https://drive.google.com/file/d/1f8ms4mK_y6ydBYM-_SMyddru7drZw4xX/view?usp=sharing)

**Notebook Visualization**: [ResNet50.ipynb](notebooks/ResNet50.ipynb)

## MobileNet

### MobileNet V1 1.0
```
make train_mobilenet1
```
[Source Code](models/mobilenet_v1.py)

- This model is trained with the new training script `train.py`

**Val Accuracy**: 63.37% (Top-1), 84.81% (Top-5)

**Training Log**: [mobilenet1-yanjiali-011119.log](logs/mobilenet1-yanjiali-011119.log)

**Pretrained Model File**: [mobilenet1-pt-yanjiali-011119.pt](https://drive.google.com/file/d/1j716uPovKbQWpnp1pKJHGWpfbrBbzpEz/view?usp=sharing)

**Notebook Visualization**: [MobileNetV1.ipynb](notebooks/MobileNetV1.ipynb)

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