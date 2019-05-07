"""Datasets."""
from .manifolds import SwissRoll, SCurve
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .stl10 import STL10
__all__ = ['SwissRoll', 'SCurve', 'MNIST', 'FashionMNIST', 'STL10']
