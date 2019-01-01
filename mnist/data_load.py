import os
import random
from os import listdir
from os.path import isfile, join
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2


class MnistDataset(Dataset):
    """
    MNIST of handwritten digits
    http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self, images_path, labels_path):
        """
        Args:
            root_dir (string): The directory with all image files (flatten).
            labels_file: The label file path
        """
        with open(images_path, 'rb') as images_f:
            b = images_f.read()
            magic = int.from_bytes(b[0:4], byteorder='big')
            count = int.from_bytes(b[4:8], byteorder='big')
            rows = int.from_bytes(b[8:12], byteorder='big')
            cols = int.from_bytes(b[12:16], byteorder='big')
            self.images = []
            for i in range(16, len(b), 28 * 28):
                image = np.asarray(list(b[i:i + 28 * 28]), dtype=np.uint8)
                image = image.reshape((28, 28))
                self.images.append(image)
        with open(labels_path, 'rb') as labels_f:
            b = labels_f.read()
            magic = int.from_bytes(b[0:4], byteorder='big')
            count = int.from_bytes(b[4:8], byteorder='big')
            self.labels = [d for d in b[8:]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
