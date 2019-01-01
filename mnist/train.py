from data_load import MnistDataset


def train():
    train_dataset = MnistDataset(
        images_path='./dataset/train-images-idx3-ubyte',
        labels_path='./dataset/train-labels-idx1-ubyte',
    )


if __name__ == "__main__":
    train()