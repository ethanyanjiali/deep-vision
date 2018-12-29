# coding: utf-8
import torch
import torch.nn as nn

# [1] Deep Residual Learning for Image Recognition https://arxiv.org/pdf/1512.03385.pdf


class ResNet34(nn.Module):
    '''
    This implements ResNet 34 layers model
    '''

    def __init__(self):
        # [conv layer]
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # [pooling layer]
        # output_size = (input_size - kernel_size) / stride + 1
        super(ResNet34, self).__init__()
        # floor[(224 - 7 + 2 * 3) / 2] + 1 = 112, it becomes 112x112x64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            # Don't use bias at conv layers any more because Batch Norm will cancel out this bias later
            bias=False,
        )
        # "We adopt batch normalization (BN) right after each convolution and before activation"[1]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # "3×3 max pool, stride 2" in Table 1 of [1]
        # ceil[(112 - 3) / 2] + 1 = 56
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)

        # this refers to Table 1 in [1]
        self.conv2x = self._make_blocks(2, 64, 64)
        self.conv3x = self._make_blocks(2, 64, 128)
        self.conv4x = self._make_blocks(2, 128, 256)
        self.conv5x = self._make_blocks(2, 256, 512)

        # in order to use FC layer, we need to downsample to 1x1x512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, 1000)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2x(x)
        x = self.conv3x(x)
        x = self.conv4x(x)
        x = self.conv5x(x)
        x = self.avgpool(x)

        # to convert batch_sizex1x1x512 to batch_sizex512
        x = x.view(x.size(0), 1 * 1 * 512)

        output = self.linear(x)
        return output

    def _make_blocks(self, num_blocks, in_channels, out_channels):
        blocks = []
        # this first block should downsample
        # "Downsampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2."[1]
        blocks.append(
            BasicBlock(in_channels, out_channels, stride=2, downsample=True))
        for i in range(1, num_blocks):
            blocks.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        # "adopt the weight initialization in [13]"[1], here [13] is He Kaiming's initialization
        # also refers to https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L116
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
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
            stride=1,
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
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
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
