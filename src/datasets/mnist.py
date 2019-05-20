"""Datasets."""
import os
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))


class MNIST(datasets.MNIST):
    """MNIST dataset."""

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ])

    def __init__(self, train=True):
        """MNIST dataset normalized."""
        super().__init__(
            BASEPATH, transform=self.transforms, train=train, download=True)

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
        normalized = normalized.view(normalized.size(0), 1, 28, 28)
        return normalized

