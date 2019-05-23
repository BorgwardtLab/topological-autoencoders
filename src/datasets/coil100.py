"""Datasets."""
import io
import glob
import os
import zipfile

import requests
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


BASEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

class COIL100Base(Dataset):
    """Rotated Objects Dataset. Base Class for downloading and loading the data"""

    url = (
        'http://www.cs.columbia.edu/CAVE/databases/'
        'SLAM_coil-20_coil-100/coil-100/coil-100.zip'
    )

    def __init__(self, root, transform=None, train=True, test_fraction=0.1,
                 seed=42):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = os.path.join(root, 'coil-100')
        self.transform = transform

        if not os.path.exists(self.root):
            self._download()

        data, labels = self._load_data()
        data_train, data_test, labels_train, labels_test = train_test_split(
            data, labels, test_size=test_fraction, stratify=labels,
            random_state=seed)
        if train:
            self.data = data_train
            self.labels = labels_train
        else:
            self.data = data_test
            self.labels = labels_test

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _download(self):
        results = requests.get(self.url)
        z = zipfile.ZipFile(io.BytesIO(results.content))
        # Get dirname of root because zip file contains the folder coil-100
        z.extractall(os.path.dirname(self.root))

    def _load_data(self):
        filelist = glob.glob(self.root + '/*.png')
        # labels are sorted according to filelist:
        labels = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
        labels = [int(i) for i in labels.values]
        data = []
        for filename in filelist:
            im = mpimg.imread(filename)
            data.append(im)
        return np.stack(data), np.array(labels)


class COIL(COIL100Base):
    """Rotated Objects Dataset."""

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            # (0.1580247, 0.1580247, 0.13644657),
            # (0.28469974, 0.22121184, 0.19787617)
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    def __init__(self, train=True):
        """COIL100 dataset normalized."""
        super().__init__(
            BASEPATH, transform=self.transforms, train=train)

    def inverse_normalization(self, normalized):
        """Inverse the normalization applied to the original data.

        Args:
            x: Batch of data

        Returns:
            Tensor with normalization inversed.

        """
        normalized = 0.5 * (normalized + 1)
        return normalized

