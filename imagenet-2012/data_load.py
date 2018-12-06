import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.image as mpimg
import cv2


class ImageNet2012Dataset(Dataset):
    """
    ImageNet LSVRC 2012 dataset.
    http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
    """

    def __init__(self, root_dir, labels_file):
        """
        Args:
            root_dir (string): The directory with all image files (flatten).
        """
        self.root_dir = root_dir
        self.tranform = transforms.Compose([
            Rescale(255),
            RandomCrop(224),
            ToTensor(),
        ])
        self.label_to_idx = {}
        self.idx_to_name = {}
        with open(labels_file, 'r') as f:
            line = fp.readline()
            idx = 0
            while line:
                line = line.strip()
                parts = line.split(' ')
                label = parts[0]
                name = "".join(parts[1:])
                self.label_to_id[label] = idx
                self.idx_to_name[idx] = name
                idx += 1
                line = fp.readline()

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                  self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # Train file name is in "n02708093_7537.JPEG" format
        # Val file name is in "n15075141_ILSVRC2012_val_00047144.JPEG" format
        parts = image_name.split('_')
        annotation = self.label_to_idx([parts[0][1:]])
        sample = {'image': image, 'annotation': annotation}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        return {'image': img, 'annotation': annotation}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]

        return {'image': image, 'annotation': annotation}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']

        # if image has no grayscale color channel, add one
        if (len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {
            'image': torch.from_numpy(image),
            'annotation': torch.from_numpy(annotation)
        }