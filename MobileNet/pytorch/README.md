# ImageNet ILSVRC2012 - PyTorch

This directory contains the PyTorch code to replicate MobileNet with public ImageNet ILSVRC2012 dataset

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

## MobileNetV1

### MobileNetV1 1.0
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