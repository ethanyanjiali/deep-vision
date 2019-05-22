from tensorflow.keras.layers import Activation, AvgPool2D, Conv2D, Dense, Flatten
from tensorflow.keras import Model, Input

# [1] Gradient-Based Learning Applied to Document Recognition http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf


def LeNet5(input_shape):
    # The architecture refers to Figure 2 in [1]
    # "Layer C1 is a convolution layer with 6 feature maps. Each unit in each feature map is connected
    # to a 5x5 neighborhood in the input"[1]
    # "the squashing function used in our Convolutional Networks is f(a)=tanh(Sa)"[1]
    x = Input(input_shape)
    conv1 = Conv2D(6, 5, strides=1, padding='valid', activation='tanh')(x)
    # "The receptive eld of each unit is a 2 by 2 area in the previous layer's corresponding feature map
    # Each unit computes the average of its four inputs"[1]
    avg1 = AvgPool2D(2, 2)(conv1)
    # "multiplies it by a trainable coeffcient adds a trainable bias and
    # passes the result through a sigmoid function"[1]
    sigmoid1 = Activation('sigmoid')(avg1)
    conv2 = Conv2D(
        16, 5, strides=1, padding='valid', activation='tanh')(sigmoid1)
    avg2 = AvgPool2D(2, 2)(conv2)
    sigmoid2 = Activation('sigmoid')(avg2)
    # "Layer C5 is a convolutional layer with 120 feature maps"[1]
    conv3 = Conv2D(
        120, 5, strides=1, padding='valid', activation='tanh')(sigmoid2)
    flatten = Flatten()(conv3)
    # "Layer F6 contains 84 units"[1]
    dense1 = Dense(84, activation='tanh')(flatten)
    # The original implementation uses RBF but we will use a 10 way softmax here for simplicity
    dense2 = Dense(10, activation='softmax')(dense1)
    # Define the mode
    model = Model(inputs=x, outputs=dense2)
    return model
