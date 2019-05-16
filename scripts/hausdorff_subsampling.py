#!/usr/bin/env python3
#
# hausdorff_subsampling.py: investigates the behaviour of the Hausdorff
# distance under random subsampling.


import numpy as np
import sys


def diameter(X):
    '''
    Calculates the diameter of a given set of points and returns it.
    '''

    # Calculate pairwise distance matrix of all points; this is required
    # as a prerequisite of the diameter calculation.
    D = np.sum((X[None, :] - X[:, None])**2, -1)**0.5

    return np.amax(D)


def hausdorff_distance(X, Y):
    '''
    Calculates the Hausdorff distance between two finite metric spaces,
    i.e. two finite point clouds.
    '''

    d_x_Y = 0

    for x in X:
        d_x_y = sys.float_info.max 
        for y in Y:
            d = np.linalg.norm(x - y, ord=None)
            d_x_y = min(d_x_y, d)

        d_x_Y = max(d_x_Y, d_x_y)

    d_y_X = 0

    for y in Y:
        d_y_x = sys.float_info.max 
        for x in X:
            d = np.linalg.norm(x - y, ord=None)
            d_y_x = min(d_y_x, d)

        d_y_X = max(d_y_X, d_y_x)

    print(d_x_Y, d_y_X)

    return max(d_x_Y, d_y_X)


if __name__ == '__main__':

    np.random.seed(42)

    n_points = 100
    d = 2
    n_subsamples = 50
    m = 10

    X = np.random.normal(size=(n_points, d))
    X_diam = diameter(X)

    for _ in range(n_subsamples):
        Y = X[np.random.choice(X.shape[0], m)]
        Y_diam = diameter(Y)

        print(hausdorff_distance(X, Y))
