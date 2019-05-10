"""Datasets."""
import os
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class STL10(datasets.STL10):
    """STL10 dataset."""

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def __init__(self, train=True):
        """STL10 dataset normalized."""
        if train:
            split='train+unlabeled' #'train'
        else:
            raise ValueError('Currently only train split implemented')

        super().__init__(
            BASEPATH, transform=self.transforms, split=split, download=True)

    @staticmethod
    def inverse_normalization(normalized):
        """Inverse the normalization applied to the original data.

        Args:
            x: Batch of data

        Returns:
            Tensor with normalization inversed.

        """
        normalized = 0.5 * (normalized + 1)
        normalized = normalized.clamp(0, 1)
        normalized = normalized.view(normalized.size(0), 3, 96, 96)
        return normalized

