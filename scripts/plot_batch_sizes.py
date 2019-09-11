"""Create Plots from parameter evaluation file"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)

    args = parser.parse_args()
    path = args.path + 'parameter_evaluations.csv'
    data = pd.read_csv(path)
    
    x = data['hyperparam_optimization_objective']
    y = data['batch_size']
    plt.scatter(y, x, marker='o')
    plt.xlabel('batch size')
    plt.ylabel('Hypersearch Obj (KL_0.1)') 
    plt.savefig(path + 'plot_batch_size.png') 

if __name__ == '__main__':
    main()
