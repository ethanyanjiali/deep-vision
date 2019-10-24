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

    def __init__(self, images_path, labels_path, mean, std):
        """
        Args:
            images_path (string): the path to the images idx file
            labels_path (string): the path to the labels idx file
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
                # convert to 28x28 2d array
                image = np.reshape(image, (28, 28))
                # pad zeros 28x28 -> 32x32
                image = np.pad(image, ((2, 2), (2, 2)), 'constant')
                # add one channel as color channel at first to match Tensor C x H x W
                image = np.reshape(image, (1, 32, 32))
                # convert ndarray to tensor
                image = torch.from_numpy(image).float()
                # normalize the image input using gien mean and std
                image = F.normalize(image, mean, std)
                self.images.append(image)
        with open(labels_path, 'rb') as labels_f:
            b = labels_f.read()
            magic = int.from_bytes(b[0:4], byteorder='big')
            count = int.from_bytes(b[4:8], byteorder='big')
            self.labels = [d for d in b[8:]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
        }