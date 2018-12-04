import torch
import torch.nn as nn
import torch.nn.functional as Func
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as Init

# [1] https://arxiv.org/pdf/1404.5997.pdf


class AlexNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # formula
        # conv layer
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # padding = ((output_size - 1) * stride) + kernel_size - input_size) / 2
        # pooling layer
        # output_size = (input_size - kernel_size) / stride + 1
        # where input_size and output_size are the square image side length

        # "The first convolutional layer filters the 224×224×3 input image with
        # 96 kernels of size 11×11×3 with a stride of 4 pixels."[1]
        # Also from [1]Fig.2, next layer is 55x55x48
        # hence padding = ((55 - 1) * 4 + 11 - 224) / 2 = 2
        # to verify, output = (224 - 11 + 2 * 2) / 4 + 1 = 55
        self.conv1 = nn.Conv2d(3, 48, 11, stride=4, padding=2)

        # From Fig.2 in [1], there's a maxpooling layer after first conv layer
        # Also from Fig.2 in [1], the pooling reduces dimension from 55x55 to 27x27
        # hence it's likely that they uses overlapping pooling kernel=3, stride=2
        # to verify, output_size = (55 - 3) / 2 + 1 = 27
        self.pooling1 = nn.MaxPool2d(3, 2)

        # "The second convolutional layer takes as input the (response-normalized
        # and pooled) output of the first convolutional layer and filters it with
        # 256 kernels of size 5 × 5 × 48."[1]

        # To achive an output size of 27, we need a combination of stride = 2 and padding = 1
        # output_size = (27 - 5 + 2 * 1) / 2 + 1 = 27
        self.conv2 = nn.Conv2d(48, 128, 5, stride=4, padding=1)

        # "The third, fourth, and fifth convolutional layers are connected to one another
        # without any intervening pooling or normalization layers"[1]
        # Also from Fig.2 in [1], next layer is 13x13x192
        # To achive an output size of 27, we need a combination of stride = 2 and padding = 1
        # output_size = (55 - 5 + 2 * 1) / 2 + 1 = 27
        self.conv3 = nn.Conv2d(48, 128, 5, stride=4, padding=1)

    def forward(self, x):
        return x
