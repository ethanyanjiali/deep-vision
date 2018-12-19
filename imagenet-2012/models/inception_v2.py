# coding: utf-8
import torch
import torch.nn as nn

# [1] https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf


class InceptionV2(nn.Module):
    '''
    This implements Inception V2, aka GoogLeNet, that described in [1]
    '''

    def __init__(self):
        super(InceptionV2, self).__init__()
        # Refer to Table 1 in [1] for the parameter below
        self.conv7x7 = nn.Conv2d(3, 64, 7, stride=2)
        self.maxpool = nn.MaxPool2d(3, 2)
        self.lrn1 = nn.LocalResponseNorm(64)
        self.conv3x3 = nn.Conv2d(64, 64, 1, stride=1)
        self.conv1x1 = nn.Conv2d(64, 192, 3, stride=1)
        self.lrn2 = nn.LocalResponseNorm(192)
        self.inception_3a = InceptionModule()
        self.inception_3b = InceptionModule()
        self.inception_4a = InceptionModule()
        self.inception_4b = InceptionModule()
        self.inception_4c = InceptionModule()
        self.inception_4d = InceptionModule()
        self.inception_4e = InceptionModule()
        self.inception_5a = InceptionModule()
        self.inception_5b = InceptionModule()

    def forward(self, x):
        pass


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        # From [1] Figure 3, we can see that inception modules are same
        # For each module, it's consist of four branches, and DepthConcat together in the end
        self.branch1_conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.branch2_conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.branch2_conv3x3 = nn.Conv2d(out_channels, out_channels, 3)
        self.branch3_conv5x5 = nn.Conv2d(in_channels, out_channels, 5)
        self.branch3_conv1x1 = nn.Conv2d(out_channels, out_channels, 1)
        self.branch4_maxpool = nn.MaxPool2d(3, 1)
        self.branch4_conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        branch1 = self.branch1_conv1x1(x)

        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_conv3x3(branch2)

        branch3 = self.branch1_conv5x5(x)
        branch3 = self.branch1_conv1x1(branch3)

        branch4 = self.branch4_maxpool(x)
        branch4 = self.branch4_conv1x1(branch4)

        outputs = [branch1, branch2, branch3, branch4]
        # this indicates the concatenation happens on depth level
        return torch.cat(outputs, 1)
