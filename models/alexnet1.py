import torch
import torch.nn as nn

# [1] https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# [2] https://arxiv.org/pdf/1404.5997.pdf
# [3] http://cs231n.github.io/convolutional-networks


class AlexNet(nn.Module):
    '''
    This is a simplified version of AlexNet from [1]. I just defined a single tower structure here.
    '''

    def __init__(self):
        super(Net, self).__init__()
        # formula
        # conv layer
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # padding = ((output_size - 1) * stride) + kernel_size - input_size) / 2
        # pooling layer
        # output_size = (input_size - kernel_size) / stride + 1
        # where input_size and output_size are the square image side length
    

        self.features = nn.Sequential(
            # "The first convolutional layer filters the 224×224×3 input image with
            # 96 kernels of size 11×11×3 with a stride of 4 pixels."[1]
            # Also from [1]Fig.2, next layer is 55x55x48, output channels is 48
            # hence padding = ((55 - 1) * 4 + 11 - 224) / 2 = 2
            # to verify, output = (224 - 11 + 2 * 2) / 4 + 1 = 55
            nn.Conv2d(3, 96, 11, stride=4, padding=2),

            # The ReLU non-linearity is applied to the output of every convolutional
            # and fully-connected layer.
            nn.ReLU(inplace=True),

            # From Fig.2 in [1], there's a maxpooling layer after first conv layer
            # Also from Fig.2 in [1], the pooling reduces dimension from 55x55 to 27x27
            # hence it's likely that they uses overlapping pooling kernel=3, stride=2
            # to verify, output_size = (55 - 3) / 2 + 1 = 27
            nn.MaxPool2d(3, 2),

            # "The second convolutional layer takes as input the (response-normalized
            # and pooled) output of the first convolutional layer and filters it with
            # 256 kernels of size 5 × 5 × 48."[1]
            # From Fig.2 in [1], output channels is 128
            # To keep dimension same as 27, we can infer that stride = 2, padding = 1
            # output_size = (27 - 5 + 2 * 2) / 1 + 1 = 27
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            # From Fig.2 in [1], there's a maxpooling layer after second conv layer
            # Also from Fig.2 in [1], the pooling reduces dimension from 27x27 to 13x13
            # similar to last one, output_size = (27 - 3) / 2 + 1 = 13
            nn.MaxPool2d(3, 2),

            # "The third, fourth, and fifth convolutional layers are connected to one another
            # without any intervening pooling or normalization layers"[1]
            # Also from Fig.2 in [1], next layer is 13x13x192, and it uses a kernel size of 3
            # to keep dimension same as 13, we can infer that stride = 1, padding = 1
            # output_size = (13 - 3 + 2 * 1) / 1 + 1 = 13
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # same as last conv layer
            # output_size = (13 - 3 + 2 * 1) / 1 + 1 = 13
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # From Fig.2 in [1], the output channels drop to 128
            # output_size = (13 - 3 + 2 * 1) / 1 + 1 = 13
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # there's another pooling layer after 5th conv layer from Fig.2 in [1]
            # output_size = (13 - 3) / 2 + 1 = 6
            nn.MaxPool2d(3, 2),
        )

        self.classifier = nn.Sequential(
            # "We use dropout in the first two fully-connected layers of Figure 2.
            # Without dropout, our network exhibits substantial overfitting.
            # Dropout roughly doubles the number of iterations required to converge."[1]
            # "...consists of setting to zero the output of each hidden neuron with probability 0.5"[1]
            nn.Dropout(p=0.5),

            # From Fig.2 in [1], the frist FC layer has 2048 activations
            nn.Linear(6 * 6 * 256, 2048),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            # From Fig.2 in [1], the second FC layer also has 2048 activations
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),

            # From Fig.2 in [1], the last FC layer output 1000 classes
            nn.Linear(2048, 1000),
        )
        

    def forward(self, x):
        x = self.features(x)

        # flatten the output from conv layers, but keep batch size
        x = x.view(x.size(0), 6 * 6 * 256)

        x = self.classifier(x)

        return x
