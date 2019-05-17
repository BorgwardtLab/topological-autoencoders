#!/usr/bin/env python3
#
# hausdorff_subsampling.py: investigates the behaviour of the Hausdorff
# distance under random subsampling.


import numpy as np
import sys

import matplotlib.pyplot as plt


def diameter(X):
    '''
    Calculates the diameter of a given set of points and returns it.
    '''

    # Calculate pairwise distance matrix of all points; this is required
    # as a prerequisite of the diameter calculation.
    D = np.sum((X[None, :] - X[:, None])**2, -1)**0.5

    return np.amax(D)


def pairwise_distances(X):
    '''
    Calculates pairwise distance matrix of a given finite metric space and
    returns it in the form of a vector.
    '''

    D = np.sum((X[None, :] - X[:, None])**2, -1)**0.5
    return D.ravel()


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

    return max(d_x_Y, d_y_X)


if __name__ == '__main__':

    np.random.seed(42)

    n_points = 200
    d = 100
    n_subsamples = 100
    m = 5

    X = np.random.normal(size=(n_points, d))
    X_diam = diameter(X)

    X_pairwise_distances = pairwise_distances(X)

    X_mean_distance = np.mean(X_pairwise_distances)
    X_mean_distance_upper_bound = np.sqrt(2 * d)

    hausdorff_distances = []
    Y_diameters = []

    for _ in range(n_subsamples):
        Y = X[np.random.choice(X.shape[0], m)]
        Y_diam = diameter(Y)

        Y_diameters.append(Y_diam)
        hausdorff_distances.append(hausdorff_distance(X, Y))

    plt.hist(X_diam - np.array(Y_diameters), bins=20)
    plt.show()

    #plt.hist(hausdorff_distances, bins=10)
    #plt.axvline(np.mean(hausdorff_distances), c='k')
    #plt.axvline(X_diam * m / n_points, c='r')
    #plt.axvline(X_diam, c='k', linestyle='dashed')

    #plt.hist(means, bins=50)
    #plt.axvline(np.mean(means), c='r')
    #plt.axvline(X_mean_distance, c='k')
    #plt.axvline(X_mean_distance_upper_bound, c='k', linestyle='dashed')
    plt.show()
