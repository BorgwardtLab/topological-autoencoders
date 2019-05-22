"""Datasets."""
from .manifolds import SwissRoll, SCurve, Spheres
from .mnist import MNIST
from .fashion_mnist import FashionMNIST
from .stl10 import STL10
from .coil100 import COIL100
__all__ = ['SwissRoll', 'SCurve', 'Spheres', 'MNIST', 'FashionMNIST', 'STL10',
           'COIL100']
