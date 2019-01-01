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
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {
            'image': torch.from_numpy(image).float(),
            'label': label,
        }


class Normalize(object):
    """Normalize the image by given pre-calculated mean and std"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, annotation = sample['image'], sample['label']

        return {
            'image': F.normalize(image, self.mean, self.std),
            'label': label,
        }