"""Manifold datasets."""
import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class SwissRoll(Dataset):
    def __init__(self, train=True, n_samples=6000, noise=0.1,
                 test_fraction=0.1, seed=42):
        super().__init__()
        self.n_samples = n_samples
        self._rnd = np.random.RandomState(seed)
        self.n_samples = n_samples
        all_data = make_swiss_roll(n_samples, noise, seed)[0]
        all_data = all_data.astype(np.float32)
        self.train_data, self.test_data = train_test_split(
            all_data, test_size=test_fraction, random_state=self._rnd)
        self.data = self.train_data if train else self.test_data

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)


class SCurve(Dataset):
    def __init__(self, train=True, n_samples=6000, noise=0.1,
                 test_fraction=0.1, seed=42):
        super().__init__()
        self._rnd = np.random.RandomState(seed)
        self.n_samples = n_samples
        all_data = make_s_curve(n_samples, noise, self._rnd)[0]
        all_data = all_data.astype(np.float32)
        self.train_data, self.test_data = train_test_split(
            all_data, test_size=test_fraction, random_state=self._rnd)
        self.data = self.train_data if train else self.test_data

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)
