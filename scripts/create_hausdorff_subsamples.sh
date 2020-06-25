#!/bin/sh
#
# Creates a set of Hausdorff distances, subsampled to show the
# convergence speed of the distance.

python3 hausdorff_subsampling.py -d 2 5 10 -n 100 -s 10 > Hausdorff_subsampling.txt
