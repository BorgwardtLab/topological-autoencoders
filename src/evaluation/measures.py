'''
Utility functions for calculating dimensionality reduction quality
measures for evaluating the latent space.
'''


import numpy as np


def pairwise_distances(X):
    '''
    Calculates pairwise distance matrix of a given data matrix and
    returns said matrix.
    '''

    D = np.sum((X[None, :] - X[:, None])**2, -1)**0.5
    return D

np.random.seed(42)
X = np.random.normal(size=(10, 2))
print(X)
