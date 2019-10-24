import tensorflow as tf

from models.mobilenet_v1 import MobileNetV1

print(tf.__version__)


def main():
    model = MobileNetV1(input_shape=(224, 224, 3))

    # Trainable params: 4,242,856
    model.summary()


if __name__ == "__main__":
    main()