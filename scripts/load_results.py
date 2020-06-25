import pandas as pd
import numpy as np
import json
import glob
from IPython import embed
import sys
from collections import defaultdict 
import collections

def nested_dict():
    return collections.defaultdict(nested_dict)

path_comp = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/fit_competitor/repetitions'
path_ae =  '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/train_model/repetitions'

#run_path='/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/fit_competitor/repetitions/rep1/CIFAR/PCA'

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
    if 'TopoRegEdge' in model:
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
                    embed()
                else:
                   #write mean and std into exp dict:
                    experiment[dataset][model][key]['mean'] = rep_vals.mean()
                    experiment[dataset][model][key]['std'] = rep_vals.std()
                    #Format mean +- std in experiment_stats dict
                    mean = rep_vals.mean()
                    std = rep_vals.std()
                    experiment_stats[dataset][model][key] = f'{mean:1.8f}' + ' \pm ' + f'{std:1.8f}' 

for dataset in datasets:
    df = pd.DataFrame.from_dict(experiment_stats[dataset], orient='index') 
    df.to_latex(f'tex/{dataset}_table_8_digits.tex')
 

#convert to df: df = pd.DataFrame.from_dict(experiment, orient='index') 
# format is then df['Vanilla']['CIFAR']['test_mean_mrre']['mean']
    



      

 
    
