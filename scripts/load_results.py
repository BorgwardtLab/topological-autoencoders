import pandas as pd
import numpy as np
import json
import glob
from IPython import embed

#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/hyperparameter_search/real_world/COIL/Vanilla/model_runs/1/run.json'
#path =  '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/hyperparameter_search/dimensionality_reduction'  
#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/train_model/best_runs'
#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/hyperparameter_search/real_world/FashionMNIST/Collected_runs/TSNE_prelim'
#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/hyperparameter_search/real_world/MNIST/Collected_runs'
#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/fit_competitor/best_runs/MNIST/UMAP'
#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/train_model/best_runs/MNIST/Vanilla'
#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/fit_competitor/best_runs/MNIST/PCA'
#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/train_model/best_runs/MNIST/Vanilla'
##path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/train_model/best_runs/MNIST/TopoRegEdgeSymmetric'
#for testing we take a single run.json
path='/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/hyperparameter_search/real_world/FashionMNIST/TSNE/model_runs/1'

#filelist= [path]
filelist = glob.glob(path + '/**/run.json', recursive=True)
print(filelist)

used_measures = ['kl_global_', 'rmse', 'mean_mrre', 'mean_continuity', 'mean_trustworthiness', 'reconstruction']

#list of flat dicts 
results = []
for filename in filelist:  
    split = filename.split('/')
    dataset = split[-5] #TODO: change to -3 and -2 for real results!
    model = split[-4]
    embed()
 
    if ('VAE' or 'Isomap') in model: #remove old trash run
        continue
    
    #nice name for proposed method:
    if 'TopoRegEdge' in model:
        model = 'TopoAE (proposed)'
    
    with open(filename, 'rb') as f:
        data = json.load(f)
    embed() 
    if 'result' not in data.keys():
        continue
        
    result_keys = list(data['result'].keys())
    #used_keys = [key for key in result_keys if 'test_density_kl_global_' in key]
    used_keys = [key for key in result_keys if any([measure in key for measure in used_measures])]

    #Create dict of results of current experiment (given dataset and model)
    experiment = {}
    experiment['dataset'] = dataset
    experiment['model'] = model
    embed() 
    #fill eval measures into experiments dict:
    for key in used_keys:
        if key in data['result'].keys():
            experiment[key] = data['result'][key]

    results.append(experiment)
    embed()

    df = pd.DataFrame(results)
    df.to_latex('test_table.tex')
     



      

 
    
