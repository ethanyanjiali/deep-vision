# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# [1] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
# https://arxiv.org/abs/1704.04861


class MobileNetV1(nn.Module):
    '''
    This implements MobileNet that introduced in [1]
    alpha: width multiplier - model shrinking hyperparameter 
    '''

    # "The role of the width multiplier α is to thin a network uniformly at each layer."[1]
    def __init__(self, alpha=1):
        super(MobileNetV1, self).__init__()
        # [conv layer]
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # [pooling layer]
        # output_size = (input_size - kernel_size) / stride + 1

        self.alpha = alpha

        self.features = nn.Sequential(
            # output (224 - 3 + 2 * 1) / 2 + 1 = 112
            # 112x112x32
            nn.Conv2d(3, alpha * 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # dw output (112 - 3 + 2 * 1) / 1 + 1 = 112
            # pw output (112 - 1 + 2 * 0) / 1 + 1 = 112
            DepthwiseSeparableConv(
                alpha * 32, alpha * 64, dw_stride=1, pw_stride=1),
            # dw output (112 - 3 + 2 * 1) / 2 + 1 = 56
            # pw output (56 - 1 + 2 * 0) / 1 + 1 = 56
            DepthwiseSeparableConv(
                alpha * 64, alpha * 128, dw_stride=2, pw_stride=1),
            # dw output (56 - 3 + 2 * 1) / 1 + 1 = 56
            # pw output (56 - 1 + 2 * 0) / 1 + 1 = 56
            DepthwiseSeparableConv(
                alpha * 128, alpha * 128, dw_stride=1, pw_stride=1),
            # dw output (56 - 3 + 2 * 1) / 2 + 1 = 28
            # pw output (28 - 1 + 2 * 0) / 1 + 1 = 28
            DepthwiseSeparableConv(
                alpha * 128, alpha * 256, dw_stride=2, pw_stride=1),
            # dw output (28 - 3 + 2 * 1) / 1 + 1 = 28
            # pw output (28 - 1 + 2 * 0) / 1 + 1 = 28
            DepthwiseSeparableConv(
                alpha * 256, alpha * 256, dw_stride=1, pw_stride=1),
            # dw output (28 - 3 + 2 * 1) / 2 + 1 = 14
            # pw output (14 - 1 + 2 * 0) / 1 + 1 = 14
            DepthwiseSeparableConv(
                alpha * 256, alpha * 512, dw_stride=2, pw_stride=1),
            #
            # 5x same depthwise separable convolution
            #
            # dw output (28 - 3 + 2 * 1) / 1 + 1 = 14
            # pw output (14 - 1 + 2 * 0) / 1 + 1 = 14
            DepthwiseSeparableConv(
                alpha * 512, alpha * 512, dw_stride=1, pw_stride=1),
            DepthwiseSeparableConv(
                alpha * 512, alpha * 512, dw_stride=1, pw_stride=1),
            DepthwiseSeparableConv(
                alpha * 512, alpha * 512, dw_stride=1, pw_stride=1),
            DepthwiseSeparableConv(
                alpha * 512, alpha * 512, dw_stride=1, pw_stride=1),
            DepthwiseSeparableConv(
                alpha * 512, alpha * 512, dw_stride=1, pw_stride=1),
            # dw output (14 - 3 + 2 * 1) / 2 + 1 = 7
            # pw output (7 - 1 + 2 * 0) / 1 + 1 = 7
            DepthwiseSeparableConv(
                alpha * 512, alpha * 1024, dw_stride=2, pw_stride=1),
            DepthwiseSeparableConv(
                alpha * 1024, alpha * 1024, dw_stride=2, pw_stride=1),
            # average pooling
            # output 1x1x1024
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.linear = nn.Linear(alpha * 1024, 1000)

    def forward(self, x):
        x = self.conv(x)
        x = self.features(x)
        x = x.view(x.size(0), 1 * 1 * self.alpha * 1024)
        x = self.linear(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dw_stride, pw_stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.dw = DepthwiseConv(in_channels, out_channels, stride=dw_stride)
        self.pw = PointwiseConv(out_channels, out_channels, stride=pw_stride)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
            stride=stride,
            # 'groups' is used to control how many depthwise filters
            # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
            groups=in_channels,
            bias=False,
        )
        # "All layers are followed by a batchnorm and ReLU nonlinearity with the
        # exception of the final fully connected layer which has no nonlinearity
        # and feeds into a softmax layer for classification."[1]
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            bias=False,
        )
        # "All layers are followed by a batchnorm and ReLU nonlinearity with the
        # exception of the final fully connected layer which has no nonlinearity
        # and feeds into a softmax layer for classification."[1]
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x