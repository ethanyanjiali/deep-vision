# Classification

PyTorch / Tensorflow / Keras implementation of common image classification models and training scripts categorized by datasets. If you think my work is helpful, please ⭐star⭐ this repo. If you have any questions regarding the code, feel free create an issue.

- [<img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png?raw=true" height="48">](https://github.com/pytorch/pytorch)
- [<img src="https://www.tensorflow.org/images/tf_logo_transp.png" height="48">](https://github.com/tensorflow/tensorflow)
- [<img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" height="48">](https://github.com/keras-team/keras) 

## Datasets

- [ImageNet ILSVRC2012](imagenet-2012) 
    - AlexNet
    - VGG
    - GoogLeNet, Inception
    - ResNet
- [CIFAR-10](cifar-10)
- [MNIST](mnist)
    - LeNet

## Disclaimer

- This repo is mainly for study purpose. Hence I write the code in a readable and understandable way, but may not be scalable and reusable. I've also added comments and referrence for those catches I ran into during replication.
- I'm not a researcher so don't have that much of time to tune the training and achieve the best benchmark. If you are looking for pre-trained models for transfer learning, there're some good ones from [PyTorch torchvision](https://pytorch.org/docs/stable/torchvision/models.html) or [TensorFlow slim](https://github.com/tensorflow/models/tree/master/research/slim).