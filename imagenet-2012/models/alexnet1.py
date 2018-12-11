import torch
import torch.nn as nn

# [1] https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# [2] http://cs231n.github.io/convolutional-networks
# [3] https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/


class AlexNet1(nn.Module):
    '''
    This implements the original AlexNet in one tower structure, hence the parameters are doubled
    '''

    def __init__(self):
        super(AlexNet1, self).__init__()
        # formula
        # [conv layer]
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # padding = ((output_size - 1) * stride) + kernel_size - input_size) / 2
        # [pooling layer]
        # output_size = (input_size - kernel_size) / stride + 1
        # where input_size and output_size are the square image side length
    

        self.features = nn.Sequential(
            # "The first convolutional layer filters the 224×224×3 input image with
            # 96 kernels of size 11×11×3 with a stride of 4 pixels."[1]
            # Also from [1]Fig.2, next layer is 55x55x48, output channels is 48. (I use 96=2x48 here)
            # hence padding = ((55 - 1) * 4 + 11 - 224) / 2 = 2
            # to verify, output = (224 - 11 + 2 * 2) / 4 + 1 = 55
            nn.Conv2d(3, 96, 11, stride=4, padding=2),

            # The ReLU non-linearity is applied to the output of every convolutional
            # and fully-connected layer.
            nn.ReLU(inplace=True),

            # "The second convolutional layer takes as input the (response-normalized
            # and pooled) output of the first convolutional layer"
            nn.LocalResponseNorm(96),

            # From Fig.2 in [1], there's a maxpooling layer after first conv layer
            # Also from Fig.2 in [1], the pooling reduces dimension from 55x55 to 27x27
            # hence it's likely that they uses overlapping pooling kernel=3, stride=2
            # to verify, output_size = (55 - 3) / 2 + 1 = 27
            nn.MaxPool2d(3, 2),

            # "The second convolutional layer takes ... with 256 kernels of size 5 × 5 × 48."[1]
            # From Fig.2 in [1], output channels is 128. (I use 256=2x128 here)
            # To keep dimension same as 27, we can infer that stride = 2, padding = 1
            # output_size = (27 - 5 + 2 * 2) / 1 + 1 = 27
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            # "The third convolutional layer has 384 kernels of size 3 × 3 ×
            # 256 connected to the (normalized, pooled) outputs of the second convolutional layer"[1]
            # Since the output of second layer is 256, the normalized layer input should be 256 here as well
            nn.LocalResponseNorm(256),

            # From Fig.2 in [1], there's a maxpooling layer after second conv layer
            # Also from Fig.2 in [1], the pooling reduces dimension from 27x27 to 13x13
            # similar to last one, output_size = (27 - 3) / 2 + 1 = 13
            nn.MaxPool2d(3, 2),

            # "The third, fourth, and fifth convolutional layers are connected to one another
            # without any intervening pooling or normalization layers"[1]
            # Also from Fig.2 in [1], next layer is 13x13x192, and it uses a kernel size of 3.
            # (I use 384=2x192 here)
            # to keep dimension same as 13, we can infer that stride = 1, padding = 1
            # output_size = (13 - 3 + 2 * 1) / 1 + 1 = 13
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # same as last conv layer
            # output_size = (13 - 3 + 2 * 1) / 1 + 1 = 13
            # (I use 384=2x192 here)
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # From Fig.2 in [1], the output channels drop to 128
            # (I use 256=2x128 here)
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

            # From Fig.2 in [1], the frist FC layer has 4096 (2x2048) activations
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            # From Fig.2 in [1], the second FC layer also has 4096 activations
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # "The output of the last fully-connected layer is fed to a 1000-way softmax which produces
            # a distribution over the 1000 class labels."[1]
            nn.Linear(4096, 1000),
            # There's no softmax here because we use CrossEntropyLoss which already includes Softmax
            # https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/5
        )

        self._initialize_weights()
        

    def forward(self, x):
        x = self.features(x)

        # flatten the output from conv layers, but keep batch size
        x = x.view(x.size(0), 6 * 6 * 256)

        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        # First, when I train the model with default weight initialization, the loss went down to around 4.9 from 6.9 after 3 epochs
        # However, not matter how small the learning rate I set, it didn't go down any more
        # https://discuss.pytorch.org/t/what-is-the-default-initialization-of-a-conv2d-layer-and-linear-layer/16055/2
        # which actually refers to http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 4.6 Initializing the weights
        #
        # Hence I replaced the weights init with the same method that I used for VGG. Please see ./vgg16 for more details
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
