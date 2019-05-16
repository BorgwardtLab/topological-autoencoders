#!/usr/bin/env python3
#
# hausdorff_subsampling.py: investigates the behaviour of the Hausdorff
# distance under random subsampling.


import numpy as np


def diameter(X):
    '''
    Calculates the diameter of a given set of points and returns it.
    '''

    # Calculate pairwise distance matrix of all points; this is required
    # as a prerequisite of the diameter calculation.
    D = np.sum((X[None, :] - X[:, None])**2, -1)**0.5

    return np.amax(D)


if __name__ == '__main__':

    np.random.seed(42)

    n_points = 100
    d = 2
    n_subsamples = 50
    m = 10

    X = np.random.normal(size=(n_points, d))

    print(diameter(X))
    np.savetxt('/tmp/X.txt', X)

    for _ in range(n_subsamples):
        Y = X[np.random.choice(X.shape[0], m)]
