# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# [1] https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf


class InceptionV1(nn.Module):
    '''
    This implements Inception V2, aka GoogLeNet, that described in [1]
    '''

    def __init__(self):
        super(InceptionV1, self).__init__()
        # [conv layer]
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # [pooling layer]
        # output_size = (input_size - kernel_size) / stride + 1

        # Refer to Table 1 in [1] for the parameter below
        # input is 224x224x3 because "The size of the receptive field in our network
        # is 224×224 in the RGB color space with zero mean."[1]
        # also we can know the output size of the first conv layer is 112x112
        # hence we can infer that padding is 3
        # so taht output_size = floor((224 - 7 + 2 * 3) / 2) + 1 = 112
        self.conv7x7 = BasicConv2d(3, 64, 7, stride=2, padding=3)
        # output_size = ceil((112 - 3) / 2) + 1 = 56
        self.maxpool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.lrn1 = nn.LocalResponseNorm(64)
        # output_size = (56 - 1 + 2 * 0) / 1 + 1 = 56
        self.conv1x1 = BasicConv2d(64, 64, 1, stride=1)
        # From table 1 in [1], this conv3x3 has output of 56×56×192
        # so output_size = (56 - 3 + 2 * 1) / 1 + 1 = 56
        self.conv3x3 = BasicConv2d(64, 192, 3, stride=1, padding=1)
        # output_size = ceil((56 - 3) / 2) + 1 = 28
        self.maxpool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.lrn2 = nn.LocalResponseNorm(192)
        # here before we go into first inception module, we have a output of
        # 28x28x192

        # input is 192, output is 256 = 64 + 128 + 32 + 32
        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        # input is 256, output is 480
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        # output_size = ceil((28 - 3) / 2) + 1 = 14
        self.maxpool3 = nn.MaxPool2d(3, 2, ceil_mode=True)
        # here before we go into inception 4x, we have output of
        # 14x14x480

        # input is 480, output is 512
        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.aux1 = AuxiliaryClassifier(512)
        # input is 512, output is 512
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        # input is 512, output is 512
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        # input is 512, output is 528
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.aux2 = AuxiliaryClassifier(528)
        # input is 528, output is 832
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        # output_size = round((14 - 3) / 2) + 1 = 7
        self.maxpool4 = nn.MaxPool2d(3, 2, ceil_mode=True)
        # here before we go into inception 5x, we have output of
        # 7x7x832

        # input is 832, output is 832
        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        # input is 832, output is 1024
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        # output_size = round((7 - 7) / 2) + 1 = 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, 1000)
        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.maxpool1(x)
        x = self.lrn1(x)
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool3(x)
        x = self.inception_4a(x)
        # aux1 uses output of 4a
        if self.training and self.aux1:
            output_aux1 = self.aux1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        # aux2 uses output of 4d
        if self.training and self.aux2:
            output_aux2 = self.aux2(x)
        x = self.inception_4e(x)
        x = self.maxpool4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), 1 * 1 * 1024)

        x = self.dropout(x)
        x = self.linear(x)
        output = self.relu(x)
        # There's no softmax here because we use CrossEntropyLoss which already includes Softmax
        # https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/5
        if self.training and output_aux1 is not None and output_aux2 is not None:
            return output, output_aux1, output_aux2
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, param1, param2, param3, param4, param5,
                 parma6):
        super(InceptionModule, self).__init__()
        # From [1] Figure 3, we can see that inception modules are same
        # For each module, it's consist of four branches, and DepthConcat together in the end
        self.branch1_conv1x1 = BasicConv2d(in_channels, param1, 1, stride=1)
        self.branch2_conv1x1 = BasicConv2d(in_channels, param2, 1, stride=1)
        self.branch2_conv3x3 = BasicConv2d(
            param2, param3, 3, stride=1, padding=1)
        self.branch3_conv1x1 = BasicConv2d(in_channels, param4, 1, stride=1)
        self.branch3_conv5x5 = BasicConv2d(
            param4, param5, 5, stride=1, padding=2)
        # https://github.com/antspy/inception_v1.pytorch/blob/master/inception_v1.py#L95
        self.branch4_maxpool = nn.MaxPool2d(3, 1, padding=1)
        self.branch4_conv1x1 = BasicConv2d(in_channels, parma6, 1, stride=1)

    def forward(self, x):
        branch1 = self.branch1_conv1x1(x)

        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_conv3x3(branch2)

        branch3 = self.branch3_conv1x1(x)
        branch3 = self.branch3_conv5x5(branch3)

        branch4 = self.branch4_maxpool(x)
        branch4 = self.branch4_conv1x1(branch4)

        outputs = [branch1, branch2, branch3, branch4]
        # this indicates the concatenation happens on depth level
        return torch.cat(outputs, 1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels):
        super(AuxiliaryClassifier, self).__init__()
        self.features = nn.Sequential(
            # "The exact structure of the extra network on the side, including the auxiliary classifier, is as follows:"[1]
            # "An average pooling layer with 5×5 filter size and
            # stride 3, resulting in an 4×4×512 output for the (4a),
            # and 4×4×528 for the (4d) stage"[1]
            nn.AvgPool2d(5, 3),
            # "A 1×1 convolution with 128 filters for dimension reduction and rectified linear activation"[1]
            BasicConv2d(in_channels, 128, 1),
        )
        self.classifier = nn.Sequential(
            # A fully connected layer with 1024 units and rectified linear activation
            nn.Linear(4 * 4 * 128, 1024),
            nn.ReLU(inplace=True),
            # "A dropout layer with 70% ratio of dropped outputs."[1]
            nn.Dropout(p=0.7),
            # A linear layer with softmax loss as the classifier (predicting the same 1000 classes as the main
            # classifier, but removed at inference time).
            nn.Linear(1024, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        # this aux network is only on 4x inception, hence after avg pool, the output will be 4x4x512 or 4x4x528
        # and after 1x1 conv2d, it will all be 4x4x128
        x = x.view(x.size(0), 4 * 4 * 128)
        x = self.classifier(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        # "All the convolutions, including those inside the Inception modules, use rectified linear activation."[1]
        return F.relu(x, inplace=True)