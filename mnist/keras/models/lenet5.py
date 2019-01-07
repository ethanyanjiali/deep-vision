from keras.layers import Activation, AvgPool2D, Conv2D, Dense, Flatten
from keras.models import Sequential

# [1] Gradient-Based Learning Applied to Document Recognition http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
# [2] https://cdn-images-1.medium.com/max/1600/1*mgYrpXPI1aOLyVtIeQYfAw.png


def LeNet5(input_shape):
    # The architecture refers to Figure 2 in [1]
    model = Sequential([
        # "Layer C1 is a convolution layer with 6 feature maps. Each unit in each feature map is connected
        # to a 5x5 neighborhood in the input"[1]
        Conv2D(6, 5, strides=1, padding='valid', input_shape=input_shape),
        # "The receptive eld of each unit is a 2 by 2 area in the previous layer's corresponding feature map
        # Each unit computes the average of its four inputs"[1]
        AvgPool2D(2, 2),
        # "multiplies it by a trainable coeffcient adds a trainable bias and
        # passes the result through a sigmoid function"[1]
        Activation('sigmoid'),
        Conv2D(16, 5, strides=1, padding='valid'),
        AvgPool2D(2, 2),
        Activation('sigmoid'),
        Flatten(),
        # "Layer C5 is a convolutional layer with 120 feature maps"[1]
        # The author claims this to be a conv layer but I use FC layer here for simplicity and acheive same effect
        # "The squashing function is a scaled hyperbolic tangent"[1]
        Dense(120, activation='tanh'),
        # "Layer F6 contains 84 units"[1]
        Dense(84, activation='tanh'),
        # The original implementation uses RBF but we will use a 10 way softmax here for simplicity
        Dense(10, activation='softmax')
    ])
    return model
