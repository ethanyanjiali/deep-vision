# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# [1] MobileNetV2: Inverted Residuals and Linear Bottlenecks
# https://arxiv.org/abs/1801.04381


class MobileNetV2(nn.Module):
    '''
    This implements MobileNet that introduced in [1]
    '''

    def __init__(self):
        super(MobileNetV2, self).__init__()
        # [conv layer]
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # [pooling layer]
        # output_size = (input_size - kernel_size) / stride + 1

    def forward(self, x):
        pass