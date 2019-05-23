import pandas as pd
import numpy as np
import json
import glob
from IPython import embed

#path = '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/hyperparameter_search/real_world/COIL/Vanilla/model_runs/1/run.json'
path =  '/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/hyperparameter_search/dimensionality_reduction'  
#for testing we take a single run.json
#filelist= [path]
filelist = glob.glob(path + '/**/run.json', recursive=True)
print(filelist)

#list of flat dicts 
results = []
for filename in filelist:  
    split = filename.split('/')
    dataset = split[-4] #TODO: change to -3 and -2 for real results!
    model = split[-2]
    
    #nice name for proposed method:
    if 'TopoRegEdge' in model:
        model = 'TopoAE (proposed)'
    
    with open(filename, 'rb') as f:
        data = json.load(f)
    
    if 'result' not in data.keys():
        continue
        
    result_keys = list(data['result'].keys())
    #used_keys = [key for key in result_keys if 'test_density_kl_global_' in key]
    used_keys = [key for key in result_keys if 'mean' in key]
  
    #Create dict of results of current experiment (given dataset and model)
    experiment = {}
    experiment['dataset'] = dataset
    experiment['model'] = model
    
    #fill eval measures into experiments dict:
    for key in used_keys:
        experiment[key] = data['result'][key]

    results.append(experiment)

    df = pd.DataFrame(results)
    df.to_latex('table_real_world.tex')
     
embed() 



      

 
    
