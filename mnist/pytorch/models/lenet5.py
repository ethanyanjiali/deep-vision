# coding: utf-8
import torch
import torch.nn as nn

# [1] Gradient-Based Learning Applied to Document Recognition http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
# [2] https://cdn-images-1.medium.com/max/1600/1*mgYrpXPI1aOLyVtIeQYfAw.png


class LeNet5(nn.Module):
    '''
    This implements LeNet-5 in [1]
    '''

    def __init__(self):
        super(LeNet5, self).__init__()
        # formula
        # [conv layer]
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # padding = ((output_size - 1) * stride) + kernel_size - input_size) / 2
        # [pooling layer]
        # output_size = (input_size - kernel_size) / stride + 1
        # where input_size and output_size are the square image side length

        # The architecture refers to Figure 2 in [1]
        self.features = nn.Sequential(
            # "Layer C1 is a convolution layer with 6 feature maps. Each unit in each feature map is connected
            # to a 5x5 neighborhood in the input"[1]
            # (32 - 5 + 2 * 0) / 1 + 1 = 28
            nn.Conv2d(1, 6, 5, stride=1),
            # "The receptive eld of each unit is a 2 by 2 area in the previous layer's corresponding feature map
            # Each unit computes the average of its four inputs"
            # (28 - 2) / 2 + 1 = 14
            nn.AvgPool2d(2, stride=2),
            # multiplies it by a trainable coeffcient adds a trainable bias and
            # passes the result through a sigmoid function
            nn.Sigmoid(),
            # (14 - 5 + 2 * 0) / 1 + 1 = 10
            nn.Conv2d(6, 16, 5, stride=1),
            # (10 - 2) / 2 + 1 = 5
            nn.AvgPool2d(2, stride=2),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            # "Layer C5 is a convolutional layer with 120 feature maps"[1]
            # The author claims this to be a conv layer but I use FC layer here for simplicity and acheive same effect
            nn.Linear(5 * 5 * 16, 120),
            # "The squashing function is a scaled hyperbolic tangent"[1]
            nn.Tanh(),
            # "Layer F6 contains 84 units"[1]
            nn.Linear(120, 84),
            nn.Tanh(),
            # The original implementation uses RBF but we will use a 10 way softmax here for simplicity
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)

        # flatten the output from conv layers, but keep batch size
        x = x.view(x.size(0), 5 * 5 * 16)

        x = self.classifier(x)

        return x
