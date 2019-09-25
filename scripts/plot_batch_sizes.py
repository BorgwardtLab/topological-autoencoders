"""Create Plots from parameter evaluation file"""
import argparse
import glob
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns #; sns.set(color_codes=True)
import pandas as pd
from IPython import embed
from parse_sacred_runtime import SacredRun

def get_n_epochs(path):
    files = glob.glob(path + 'latent_epoch_*.pdf')
    return len(files)

def read_runtimes(path, n_runs=20):
    run_path = path + 'model_runs/{}/' 
    run_path_json = run_path + 'run.json'
    #get list of all paths to all run.json files:
    run_paths = [run_path.format(i) for i in np.arange(1,n_runs+1)] 
    run_paths_json = [run_path_json.format(i) for i in np.arange(1,n_runs+1)] 

    runtimes = []

    #iterate over model_runs:
    for pathname, filename in zip(run_paths, run_paths_json):
        if SacredRun.is_valid(filename):
            run = SacredRun(filename)
            n_epochs = get_n_epochs(pathname)
            run_time_per_epoch = run.runtime/ float(n_epochs)  
            runtimes.append({'run': run.run_file, 'runtime_in_seconds': run.runtime, 'n_epochs': n_epochs, 'runtime_per_epoch_in_seconds': run_time_per_epoch  })
            print(f'{run.run_file}: {run.runtime} Total, {run_time_per_epoch} per epoch')
    df = pd.DataFrame(runtimes)
    if len(df) < n_runs: 
        print(f'Not all runs were valid! Out of {n_runs}, only {len(df)} runs could be used!')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, type=str)

    args = parser.parse_args()
    
    fit_reg = True
    marker = "."
    path = args.path + 'parameter_evaluations.csv'
    data = pd.read_csv(path)
    runtimes = read_runtimes(args.path)

    batch_size = data['batch_size']
    xmin = np.min(batch_size) - 3
    xmax = np.max(batch_size) + 3
    #plt.figure()
    fig, ax = plt.subplots()

    used = ['batch_size', 'training.loss.autoencoder', 'training.loss.topo_error', 'hyperparam_optimization_objective'] 
    labels = ['Reconstruction Loss', 'Topological Loss', 'Objective (KL_0.1)', 'Runtime per epoch']
    colors = ['blue', 'orange', 'green', 'red']
    
    df = data[used]
    
    for i in np.arange(1,len(used)):
        #sns.plt.xlim(35, 100)
        sns.regplot(x='batch_size', y=used[i], data=df, label=labels[i-1], color=colors[i-1], truncate=False, ax=ax, fit_reg=fit_reg)      
        ax.set(xlabel="Batch Size", ylabel = "Loss Measures")
        ax.set_xlim(xmin, xmax)
    ax.legend(loc='upper center') #right
    
    data['runtimes'] = runtimes['runtime_per_epoch_in_seconds'].values
    #embed()

    #add runtime plot:
    #labels.append('Runtime per epoch')
    ax2 = ax.twinx()
    #color = 'red'
    #ax2.set_ylabel('Runtime per epoch (in seconds)', color) 
    #ax2.set(ylabel = "Runtime per epoch (in seconds)", color=colors[-1])
    ax2.set_ylabel('Runtime per epoch (in seconds)', color=colors[-1])
    #sns.regplot(x='batch_size', y='runtimes', data=data, label=labels[-1], truncate=False, ax=ax2, color=colors[-1], logx=False, fit_reg=fit_reg)
    order = np.argsort(data['batch_size'])
    ax2.plot(data['batch_size'][order], data['runtimes'][order], label=labels[-1], color=colors[-1])
    ax2.tick_params(axis='y', labelcolor=colors[-1])
    
    #ax2.legend(loc='upper left')
    plt.savefig('plots/' + f'plot_batch_size_vs_losses_and_runtimes.png') 
    
#    sns_plot = sns.pairplot(df)
#    sns_plot.savefig('plots/test_pairplot.png')



    #plt.savefig(path + 'plot_batch_size.png') 
    #plt.savefig('plots/' + 'plot_batch_size.png') 

if __name__ == '__main__':
    main()
