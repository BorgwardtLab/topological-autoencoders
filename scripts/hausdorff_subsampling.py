#!/usr/bin/env python3
#
# hausdorff_subsampling.py: investigates the behaviour of the Hausdorff
# distance under random subsampling.

import argparse
import sys

import numpy as np
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dimension', type=int, nargs='+', default=2)
    parser.add_argument('-n', '--n_points', type=int, default=100)
    parser.add_argument('-s', '--n_subsamples', type=int, default=20)
    parser.add_argument('-m', type=int, default=25)

    args = parser.parse_args()

    n_points = args.n_points
    d = args.dimension
    n_subsamples = args.n_subsamples
    m = args.m

    for d in args.dimension:

        print(f'# dim = {d}') 

        np.random.seed(42)

        X = np.random.normal(size=(n_points, d))
        X_diam = diameter(X)

        X_pairwise_distances = pairwise_distances(X)

        X_mean_distance = np.mean(X_pairwise_distances)
        X_mean_distance_upper_bound = np.sqrt(2 * d)

        for m in range(1, 101):

            hausdorff_distances = []
            Y_diameters = []

            for _ in range(n_subsamples):
                Y = X[np.random.choice(X.shape[0], m, replace=False)]

                Y_diameters.append(diameter(Y))
                hausdorff_distances.append(hausdorff_distance(X, Y))

            #print(np.array([X_diam] * n_subsamples), np.array(Y_diameters), hausdorff_distances)
            #print(Y_diameters - X_diam)
            #print(np.all(Y_diameters >= hausdorff_distances))

            #print(np.mean(hausdorff_distances) / np.mean(Y_diameters))

            print(m, np.mean(hausdorff_distances))

        print()
