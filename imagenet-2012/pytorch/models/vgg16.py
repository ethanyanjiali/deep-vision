# coding: utf-8
import torch
import torch.nn as nn

# [1] https://arxiv.org/pdf/1409.1556.pdf
# [2] https://arxiv.org/pdf/1502.01852.pdf

class VGG16(nn.Module):
    '''
    This implements model D with 16 weight layers in VGG paper.
    '''
    def __init__(self):
        super(VGG16, self).__init__()
        # formula
        # [conv layer]
        # output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # padding = ((output_size - 1) * stride) + kernel_size - input_size) / 2
        # [pooling layer]
        # output_size = (input_size - kernel_size) / stride + 1
        # where input_size and output_size are the square image side length

        # This model implements D in Table 1: ConvNet configurations of [1]
        # I could have make some helper function to reduce boilderplate, but i think it's 
        # much more clear by listing all layers line by line
        self.features = nn.Sequential(
            # "The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input
            # is such that the spatial resolution is preserved 
            # after convolution, i.e. the padding is 1 pixel for 3 × 3 conv. layers"[1]
            # output = (224 - 3 + 2 * 1) / 1 + 1 = 224
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            # "All hidden layers are equipped with the rectification (ReLU (Krizhevsky et al., 2012)) non-linearity"[1]
            nn.ReLU(inplace=True),
            # output = (224 - 3 + 2 * 1) / 1 + 1 = 224
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # "Max-pooling is performed over a 2 × 2 pixel window, with stride 2."[1]
            # output = (224 - 2) / 2 + 1 = 112
            nn.MaxPool2d(2, 2),

            # output = (112 - 3 + 2 * 1) / 1 + 1 = 112
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (112 - 3 + 2 * 1) / 1 + 1 = 112
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (112 - 2) / 2 + 1 = 56
            nn.MaxPool2d(2, 2),

            # output = (56 - 3 + 2 * 1) / 1 + 1 = 56 
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (56 - 3 + 2 * 1) / 1 + 1 = 56 
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (56 - 3 + 2 * 1) / 1 + 1 = 56 
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (56 - 2) / 2 + 1 = 28
            nn.MaxPool2d(2, 2),

            # output = (28 - 3 + 2 * 1) / 1 + 1 = 28
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (28 - 3 + 2 * 1) / 1 + 1 = 28
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (28 - 3 + 2 * 1) / 1 + 1 = 28
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (28 - 2) / 2  + 1 = 14
            nn.MaxPool2d(2, 2),

            # output = (14 - 3 + 2 * 1) / 1 + 1 = 14
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (14 - 3 + 2 * 1) / 1 + 1 = 14
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (14 - 3 + 2 * 1) / 1 + 1 = 14
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # output = (14 - 2) / 2 + 1 = 7
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            # "...and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5)."[1]
            nn.Dropout(p=0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            # There's no softmax here because we use CrossEntropyLoss which already includes Softmax
            # https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/5
        )
        
        # A deep network like VGG requires proper intialization to be able to converge
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)

        # flatten the output from conv layers, but keep batch size
        x = x.view(x.size(0), 7 * 7 * 512)

        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        # First, when I train the model with default weight initialization, it doesn't show any sign of convergence at all after
        # 2 epochs, the loss remain same at 6.907
        # https://discuss.pytorch.org/t/what-is-the-default-initialization-of-a-conv2d-layer-and-linear-layer/16055/2
        # which actually refers to http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf 4.6 Initializing the weights
        #
        # "The initialisation of the network weights is important, since bad initialisation can stall learning due
        # to the instability of gradient in deep nets. To circumvent this problem, we began with training
        # the configuration A (Table 1), shallow enough to be trained with random initialisation. Then, when
        # training deeper architectures, we initialised the first four convolutional layers and the last three fullyconnected
        # layers with the layers of net A (the intermediate layers were initialised randomly)"[1]
        #
        # However this would be too much hassle for us. So I refer to [2]
        # in 2.2. Initialization of Filter Weights for Rectifiers, it gives details about how to initialize weights
        # that allows for extremely deep models to converge. This has been implemented in PyTroch as kaiming_normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
