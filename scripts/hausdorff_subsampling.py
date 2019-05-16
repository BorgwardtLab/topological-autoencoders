#!/usr/bin/env python3
#
# hausdorff_subsampling.py: investigates the behaviour of the Hausdorff
# distance under random subsampling.


import numpy as np


if __name__ == '__main__':

    np.random.seed(42)

    n_points = 100
    d = 2
    n_subsamples = 50
    m = 10

    X = np.random.normal(size=(n_points, d))

    for _ in range(n_subsamples):
        Y = X[np.random.choice(X.shape[0], m)]
