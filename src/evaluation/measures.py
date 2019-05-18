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


def stress(X, Z):
    '''
    Calculates the stress measure between the data space `X` and the
    latent space `Z`.
    '''

    sum_of_squared_differences = np.square(X - Z).sum()
    sum_of_squares = np.square(Z).sum()

    return np.sqrt(sum_of_squared_differences / sum_of_squares)


def RMSE(X, Z):
    '''
    Calculates the RMSE measure between the data space `X` and the
    latent space `Z`.
    '''

    n = X.shape[0]
    sum_of_squared_differences = np.square(X - Z).sum()
    return np.sqrt(sum_of_squared_differences / n**2)


np.random.seed(42)
X = np.random.normal(size=(10, 2))
Z = np.random.normal(size=(10, 2))

print(stress(X, Z))
print(RMSE(X, Z))
