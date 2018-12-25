# coding: utf-8
import torch
import torch.nn as nn

# [1] https://arxiv.org/pdf/1512.03385.pdf


class ResNet34(nn.Module):
    '''
    This implements the original AlexNet in one tower structure, hence the parameters are doubled
    '''

    def __init__(self):
        super(ResNet34, self).__init__()

        self._initialize_weights()

    def forward(self, x):

        return x

    def _initialize_weights(self):
        # First, when I train the model with default weight initialization (Lecun's), the loss went down to around 4.9 from 6.9 after 3 epochs
        # However, not matter how small the learning rate I set, it didn't go down any more
        # https://discuss.pytorch.org/t/what-is-the-default-initialization-of-a-conv2d-layer-and-linear-layer/16055/2
        # which actually refers to http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 4.6 Initializing the weights
        #
        # Hence I replaced the weights init with the same method that I used for VGG. Please see ./vgg16 for more details
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BasicBlock, self).__init__()
        # According to Figure 5 Left in [1]
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.projection = None
        self.downsample = downsample
        if downsample:
            # "(B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions)."[1]
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.projection(identity)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        output = x + identity
        output = self.relu(output)
        return output
