#  ImageNet ILSVRC2012

The [ImageNet ILSVRC2012](https://en.wikipedia.org/wiki/ImageNet) is a subset of the large hand-labeled ImageNet dataset (10,000,000 labeled images depicting 10,000+ object categories) as training. The validation and test data for this competition will consist of 150,000 photographs, collected from flickr and other search engines, hand labeled with the presence or absence of 1000 object categories. The goal of ILSVRC 2012 competition is to estimate the content of photographs for the purpose of retrieval and automatic annotation.

## Set Up Dataset

Follow the instruction here [DATASET.md](DATASET.md) to download the ILSVRC2012 dataset first.

## Implementations

I've made the networks and training scripts in multiple frameworks. Please refer to their own directory for specific implementation.

- [PyTorch](pytorch)
    - AlexNet V1
    - AlexNet V2
    - VGG-16
    - VGG-19
    - Inception V1
    - Inception V3
    - ResNet-34
    - ResNet-50
    - ResNet-152
    - MobileNet V1
- [TensorFlow](tensorflow)
    - N/A