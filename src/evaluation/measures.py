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

def get_neighbours_and_ranks(X, k):
    '''
    Calculates the neighbourhoods and the ranks of a given space `X`,
    and returns the corresponding tuple. An additional parameter $k$,
    the size of the neighbourhood, is required.
    '''
    X = pairwise_distances(X)

    # Warning: this is only the ordering of neighbours that we need to
    # extract neighbourhoods below. The ranking comes later!
    X_ranks = np.argsort(X, axis=-1, kind='stable')

    # Extract neighbourhoods.
    X_neighbourhood = X_ranks[:, 1:k+1]

    # Convert this into ranks (finally)
    X_ranks = X_ranks.argsort(axis=-1, kind='stable')

    return X_neighbourhood, X_ranks


def trustworthiness(X, Z, k):
    '''
    Calculates the trustworthiness measure between the data space `X`
    and the latent space `Z`, given a neighbourhood parameter `k` for
    defining the extent of neighbourhoods.
    '''

    X_neighbourhood, X_ranks = get_neighbours_and_ranks(X, k)
    Z_neighbourhood, Z_ranks = get_neighbours_and_ranks(Z, k)

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


def continuity(X, Z, k):
    '''
    Calculates the continuity measure between the data space `X` and the
    latent space `Z`, given a neighbourhood parameter `k` for setting up
    the extent of neighbourhoods.

    This is just the 'flipped' variant of the 'trustworthiness' measure.
    '''

    # Notice that the parameters have to be flipped here.
    return trustworthiness(Z, X, k)

def neighbourhood_loss(X, Z, k):
    '''
    Calculates the neighbourhood loss quality measure between the data
    space `X` and the latent space `Z` for some neighbourhood size $k$
    that has to be pre-defined.
    '''

    X_neighbourhood, _ = get_neighbours_and_ranks(X, k)
    Z_neighbourhood, _ = get_neighbours_and_ranks(Z, k)

    result = 0.0
    n = X.shape[0]

    for row in range(n):
        shared_neighbours = np.intersect1d(
            X_neighbourhood[row],
            Z_neighbourhood[row]
        )

        result += len(shared_neighbours) / k

    return 1.0 - result / n


def MRRE(X, Z, k):
    '''
    Calculates the mean relative rank error quality metric of the data
    space `X` with respect to the latent space `Z`, subject to its $k$
    nearest neighbours.
    '''

    X_neighbourhood, X_ranks = get_neighbours_and_ranks(X, k)
    Z_neighbourhood, Z_ranks = get_neighbours_and_ranks(Z, k)

    n = X.shape[0]

    # First component goes from the latent space to the data space, i.e.
    # the relative quality of neighbours in `Z`.

    mrre_ZX = 0.0
    for row in range(n):
        for neighbour in Z_neighbourhood[row]:
            rx = X_ranks[row, neighbour]
            rz = Z_ranks[row, neighbour]

            mrre_ZX += abs(rx - rz) / rz

    # Second component goes from the data space to the latent space,
    # i.e. the relative quality of neighbours in `X`.

    mrre_XZ = 0.0
    for row in range(n):
        # Note that this uses a different neighbourhood definition!
        for neighbour in X_neighbourhood[row]:
            rx = X_ranks[row, neighbour]
            rz = Z_ranks[row, neighbour]

            # Note that this uses a different normalisation factor
            mrre_XZ += abs(rx - rz) / rx

    # Normalisation constant
    C = n * sum([abs(2*j - n - 1) / j for j in range(1, k+1)])
    return mrre_ZX / C, mrre_XZ / C
