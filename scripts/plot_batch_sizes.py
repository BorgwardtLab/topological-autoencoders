"""Create Plots from parameter evaluation file"""
import argparse
import numpy as np

import seaborn as sns #; sns.set(color_codes=True)

import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)

    args = parser.parse_args()

    marker = "."
    path = args.path + 'parameter_evaluations.csv'
    data = pd.read_csv(path)
    
#    fig, ax = plt.subplots(3,1, sharex=True)
#    x = data['hyperparam_optimization_objective']
#    y = data['batch_size']
#    ax[0].scatter(y,x, marker=marker)
#    #ax[0].set_xlabel('Batch size')
#    ax[0].set_ylabel('Objective (KL_0.1)') 
#    
#    x2 = data['training.loss.topo_error'] 
#    ax[1].scatter(y,x2, marker=marker)
#    #ax[1].set_xlabel('Batch size')
#    ax[1].set_ylabel('Topological loss')
#
#    x3 = data['training.loss.autoencoder'] 
#    ax[2].scatter(y,x3, marker=marker)
#    ax[2].set_xlabel('Batch size')
#    ax[2].set_ylabel('Reconstruction loss')

    batch_size = data['batch_size']
    xmin = np.min(batch_size) - 3
    xmax = np.max(batch_size) + 3
    #plt.figure()
    fig, ax = plt.subplots()

    used = ['batch_size', 'training.loss.autoencoder', 'training.loss.topo_error', 'hyperparam_optimization_objective'] 
    labels = ['Reconstruction Loss', 'Topological Loss', 'Objective (KL_0.1)']
    df = data[used]
    
    for i in np.arange(1,len(used)):
        #sns.plt.xlim(35, 100)
        sns.regplot(x='batch_size', y=used[i], data=df, label=labels[i-1], truncate=False, ax=ax)      
        ax.set(xlabel="Batch Size", ylabel = "Loss Measures")
        ax.set_xlim(xmin, xmax)
        plt.legend()
    plt.savefig('plots/' + f'plot_batch_size_vs_losses.png') 
    
#    sns_plot = sns.pairplot(df)
#    sns_plot.savefig('plots/test_pairplot.png')



    #plt.savefig(path + 'plot_batch_size.png') 
    #plt.savefig('plots/' + 'plot_batch_size.png') 

if __name__ == '__main__':
    main()
