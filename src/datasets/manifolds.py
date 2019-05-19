"""Manifold datasets."""
import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from .topo_dataset.spheres import create_sphere_dataset

def normalize_features(data_train, data_test):
    """Normalize features to zero mean and unit variance.

    Args:
        data:

    Returns:
        (transformed_data_train, transformed_data_test)

    """
    mean = np.mean(data_train, axis=0, keepdims=True)
    std = np.std(data_train, axis=0, keepdims=True)
    transformed_train = (data_train - mean) / std
    transformed_test = (data_test - mean) / std
    return transformed_train, transformed_test


class ManifoldDataset(Dataset):
    def __init__(self, data, position, train, test_fraction, random_seed):
        train_data, test_data, train_pos, test_pos = train_test_split(
            data, position, test_size=test_fraction, random_state=random_seed)
        self.train_data, self.test_data = normalize_features(
            train_data, test_data)
        self.train_pos, self.test_pos = train_pos, test_pos
        self.data = self.train_data if train else self.test_data
        self.pos = self.train_pos if train else self.test_pos

    def __getitem__(self, index):
        return self.data[index], self.pos[index]

    def __len__(self):
        return len(self.data)


class SwissRoll(ManifoldDataset):
    def __init__(self, train=True, n_samples=6000, noise=0.05,
                 test_fraction=0.1, seed=42):
        _rnd = np.random.RandomState(seed)
        data, pos = make_swiss_roll(n_samples, noise, seed)
        data = data.astype(np.float32)
        pos = pos.astype(np.float32)
        super().__init__(data, pos, train, test_fraction, _rnd)


class SCurve(ManifoldDataset):
    def __init__(self, train=True, n_samples=6000, noise=0.05,
                 test_fraction=0.1, seed=42):
        _rnd = np.random.RandomState(seed)
        data, pos = make_s_curve(n_samples, noise, _rnd)
        data = data.astype(np.float32)
        pos = pos.astype(np.float32)
        super().__init__(data, pos, train, test_fraction, _rnd)

class Spheres(ManifoldDataset):
    def __init__(self, train=True, n_samples=500, d=100, n_spheres=11, r=5,
                test_fraction=0.1, seed=42):
        #here pos are actually class labels, just conforming with parent class!
        data, labels = create_sphere_dataset(n_samples, d, n_spheres, r, seed=seed)
        pos = labels
        data = data.astype(np.float32)
        pos = pos.astype(np.float32)
        _rnd = np.random.RandomState(seed)
        super().__init__(data, pos, train, test_fraction, _rnd)


