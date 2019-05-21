"""Datasets."""
import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class MNIST(datasets.MNIST):
    """MNIST dataset."""

    mean_channels = (0.131,)
    std_channels = (0.308,)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """MNIST dataset normalized."""
        super().__init__(
            BASEPATH, transform=self.transforms, train=train, download=True)

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data.

        Args:
            x: Batch of data

        Returns:
            Tensor with normalization inversed.

        """
        normalized = 0.5 * (normalized + 1)
        return normalized

