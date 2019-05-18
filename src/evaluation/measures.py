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

    X = pairwise_distances(X)
    Z = pairwise_distances(Z)

    sum_of_squared_differences = np.square(X - Z).sum()
    sum_of_squares = np.square(Z).sum()

    return np.sqrt(sum_of_squared_differences / sum_of_squares)


def RMSE(X, Z):
    '''
    Calculates the RMSE measure between the data space `X` and the
    latent space `Z`.
    '''

    X = pairwise_distances(X)
    Z = pairwise_distances(Z)

    n = X.shape[0]
    sum_of_squared_differences = np.square(X - Z).sum()
    return np.sqrt(sum_of_squared_differences / n**2)


def trustworthiness(X, Z, k):
    '''
    Calculates the trustworthiness measure between the data space `X`
    and the latent space `Z`, given a neighbourhood parameter `k` for
    defining the extent of neighbourhoods.
    '''

    X = pairwise_distances(X)
    Z = pairwise_distances(Z)

    # Warning: this is only the ordering of neighbours that we need to
    # extract neighbourhoods below. The ranking comes later!
    X_ranks = np.argsort(X, axis=-1, kind='stable')
    Z_ranks = np.argsort(Z, axis=-1, kind='stable')

    # Extract neighbourhoods.
    X_neighbourhood = X_ranks[:, 1:k+1]
    Z_neighbourhood = Z_ranks[:, 1:k+1]

    # Convert this into ranks (finally) in order to make the lookup
    # possible later on.
    X_ranks = X_ranks.argsort(axis=-1, kind='stable')
    Z_ranks = Z_ranks.argsort(axis=-1, kind='stable')

    result = 0.0

    # Calculate number of neighbours that are in the $k$-neighbourhood
    # of the latent space but not in the $k$-neighbourhood of the data
    # space.
    for row in range(X_ranks.shape[0]):
        missing_neighbours = np.setdiff1d(
            Z_neighbourhood[row],
            X_neighbourhood[row]
        )

        for neighbour in missing_neighbours:
            result += (X_ranks[row, neighbour] - k)

    n = X.shape[0]
    return 1 - 2 / (n * k * (2 * n - 3 * k - 1) ) * result



np.random.seed(42)
X = np.random.normal(size=(10, 2))
Z = np.random.normal(size=(10, 2))

print(stress(X, Z))
print(RMSE(X, Z))
print(trustworthiness(X, Z, 1))
