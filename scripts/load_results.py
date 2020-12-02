import pandas as pd
import numpy as np
import json
import glob
from IPython import embed
import sys
from collections import defaultdict 
import collections

def highlight_best_with_std(df, larger_is_better_dict, top=2): 
    """ actually operates on dataframes
        here, takes df with mean (to determine best), and one with std as to reformat
    """
    formats = [ [r' \first{',    ' }'],
        [r' \second{ ',               ' }']]

    for col in df.columns:
        # as pm formatting occured before, extract means:
        means = df[col].str.split(' ', n=1, expand=True)
        means[0] = means[0].astype(float)
        if larger_is_better_dict[col]:
            top_n = means[0].nlargest(2).index.tolist() 
        else:
            top_n = means[0].nsmallest(2).index.tolist() 
        rest = list(df[col].index)
        for i, best in enumerate(top_n):
            df[col][best] = formats[i][0] + f'{df[col][best]}' + formats[i][1] 
            rest.remove(best)
        
    return df


def nested_dict():
    return collections.defaultdict(nested_dict)


# CAVE: set the following paths accordingly!
outpath = 'tex_TopoPCA' #'tex'
path_comp = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/fit_competitor/repetitions'
path_ae =  '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/train_model/repetitions'

#filelist= [path]
files_comp = glob.glob(path_comp + '/**/run.json', recursive=True)
files_ae = glob.glob(path_ae + '/**/run.json', recursive=True)
filelist = files_ae + files_comp
print(filelist)

used_measures = ['kl_global_', 'rmse', '_mse', 'mean_mrre', 'mean_continuity', 'mean_trustworthiness', 'reconstruction']

#list of flat dicts 
results = []
experiment = nested_dict() #defaultdict(dict)
experiment_stats = nested_dict()

#1. Gather all results in experiment dict
datasets = []
models = []
repetitions = np.arange(1,6)
all_used_keys = []

for filename in filelist:  
    split = filename.split('/')
    repetition = int(split[-4][-1])
    dataset = split[-3] #TODO: change to -3 and -2 for real results!
    if dataset not in datasets:
        datasets.append(dataset)
    model = split[-2]

    #nice name for proposed method:
    if 'LinearAE' in model:
        model = 'TopoPCA'
    elif 'TopoRegEdge' in model:
        model = 'TopoAE (proposed)'
    if model not in models:
        models.append(model)
 
    with open(filename, 'rb') as f:
        data = json.load(f)

    run_file = filename.split('/')[-1] 
    metrics_path = filename.strip(run_file) + 'metrics.json'
    with open(metrics_path, 'rb') as f:
        metrics = json.load(f)
    # metrics['testing.reconstruction_error']['values'][-1] for accessing latest recon error..
    
    if 'result' not in data.keys():
        continue
        
    result_keys = list(data['result'].keys())
    
    #used_keys = [key for key in result_keys if 'test_density_kl_global_' in key]
    used_keys = [key for key in result_keys if any([measure in key for measure in used_measures])]

    #Update list of all keys ever used for later processing (use more speaking name for PCA test recon
    for key in used_keys:
        if key == 'test_mse':
            new_key = 'test.reconstruction_error' #better name for PCA test reconstruction error
        else:
            new_key = key
        if new_key not in all_used_keys:
            all_used_keys.append(new_key)

    #fill eval measures into experiments dict:
    for key in used_keys: #still loop over old naming, as it is stored this way in json..
        if key in data['result'].keys():
            if key == 'test_mse':
                new_key = 'test.reconstruction_error' #better name for PCA test reconstruction error
            else:
                new_key = key
            if key == 'training.reconstruction_error': #use test recon (stored in metrics)
                new_key = 'test.reconstruction_error'
                if new_key not in all_used_keys:
                    all_used_keys.append(new_key)
                test_recon = metrics['testing.reconstruction_error']['values'][-1]
                print(f'Entering into {new_key} following Test Recon: {test_recon} for {dataset}/{model}/{repetition}')
                experiment[dataset][model][new_key][repetition] = test_recon 
            else:
                experiment[dataset][model][new_key][repetition] = data['result'][key]

#2. Check that 5 repes avail and then compute mean + std
for dataset in datasets:
    for model in models:
        loaded_results = experiment[dataset][model]
        for key in all_used_keys:
            if key in loaded_results.keys(): #not all methods have recon error
                rep_vals = np.array(list(loaded_results[key].values())) 
                n_reps = len(rep_vals)
                if n_reps < 5: 
                    print(f'Less than 5 reps in exp: {dataset}/{model}/{key}')
                    #embed()
                else:
                   #write mean and std into exp dict:
                    experiment[dataset][model][key]['mean'] = rep_vals.mean()
                    experiment[dataset][model][key]['std'] = rep_vals.std()
                    #Format mean +- std in experiment_stats dict
                    mean = rep_vals.mean()
                    std = rep_vals.std()
                    if key == 'test_density_kl_global_10':
                        experiment_stats[dataset][model][key] = f'{mean:1.6f}' + ' $\pm$ ' + f'{std:1.6f}' 
                    else:
                        experiment_stats[dataset][model][key] = f'{mean:1.5f}' + ' $\pm$ ' + f'{std:1.5f}' 

col_mapping = { 'test_density_kl_global_0001':  '$\dkl_{0.001}$',
                'test_density_kl_global_001':   '$\dkl_{0.01}$',
                'test_density_kl_global_01':    '$\dkl_{0.1}$',
                'test_density_kl_global_1':     '$\dkl_{1}$', 
                'test_density_kl_global_10':    '$\dkl_{10}$', 
                'test_mean_continuity':         '$\ell$-Cont', 
                'test_mean_mrre':               '$\ell$-MRRE',  
                'test_mean_trustworthiness':    '$\ell$-Trust',
                'test_rmse':                    '$\ell$-RMSE', 
                'test.reconstruction_error':    'Data MSE' 
} 
larger_is_better = {  
'$\dkl_{0.001}$': 0,
'$\dkl_{0.01}$': 0,
'$\dkl_{0.1}$': 0,
'$\dkl_{1}$': 0, 
'$\dkl_{10}$': 0, 
'$\ell$-Cont': 1, 
'$\ell$-MRRE': 0,  
'$\ell$-Trust': 1,
'$\ell$-RMSE': 0, 
'Data MSE': 0 
}

for dataset in datasets:
    df = pd.DataFrame.from_dict(experiment_stats[dataset], orient='index')
    df = df.rename(columns=col_mapping)
    if dataset == 'Spheres':
        df['order'] = [4,5,6,3,1,2,0]
    else:
        df['order'] = [3,4,5,2,0,1]
    df = df.sort_values(by=['order']) 
    df = df.drop(columns=['order'])
    #embed()
    df = highlight_best_with_std(df, larger_is_better)
    #embed()
 
    df.to_latex(f'{outpath}/{dataset}_table_5_digits.tex', escape=False)

#convert to df: df = pd.DataFrame.from_dict(experiment, orient='index') 
# format is then df['Vanilla']['CIFAR']['test_mean_mrre']['mean']
    



      

 
    
