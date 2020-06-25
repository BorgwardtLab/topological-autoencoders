import numpy as np

import os
import sys
sys.path.append('../')

import torch

from src.datasets import CIFAR
from src.datasets import FashionMNIST
from src.datasets import MNIST
from src.datasets import Spheres

from src.evaluation.utils import get_space


def load_data(name):
    '''
    Loads data from a given data set and uses it to return a proper
    point cloud.

    :param name: Data set to load
    '''

    data_set = name(train=False)

    loader = torch.utils.data.DataLoader(
        data_set, batch_size=64, drop_last=False
    )

    data, labels = get_space(None, loader, mode='data')
    return data, labels


def save_all(data, labels, name):

    os.makedirs(f'bottleneck/{name}', exist_ok=True)

    np.savetxt(f'bottleneck/{name}/data.csv', data, delimiter=',')
    np.savetxt(f'bottleneck/{name}/labels.csv', labels, delimiter=',')

if __name__ == '__main__':
    for name, c in [
        ('CIFAR', CIFAR),
        ('FashionMNIST', FashionMNIST),
        ('MNIST', MNIST),
        ('Spheres', Spheres)
    ]:
        data, labels = load_data(c)
        save_all(data, labels, name)
