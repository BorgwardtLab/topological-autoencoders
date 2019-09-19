import pandas as pd
import numpy as np
from IPython import embed

import sys
import torch
sys.path.append('../')

from src.datasets import MNIST
from src.datasets import Spheres 
from src.evaluation.utils import get_space


def load_latents():
    """ hard coded to load MNIST, UMAP latents for now
    """
    #latent_path='/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/fit_competitor/repetitions/rep1/MNIST/UMAP/latents.csv'
    latent_path='/links/groups/borgwardt/Projects/TopoAE/topologically-constrained-autoencoder/exp_runs/fit_competitor/repetitions/rep1/Spheres/UMAP/latents.csv'
    latent_df = pd.read_csv(latent_path) # pd Df with columns: [dim0, dim1, label]
    latents = latent_df.iloc[:,:-1].values # np array of shape [samples, features (2) ]
    latents_labels = latent_df['labels'].values 
    return latents, latents_labels

def load_data():
    """ hard coded to load MNIST for now
    """ 
    #test_dataset = MNIST(train=False)
    test_dataset = Spheres(train=False)

    dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=200, pin_memory=True,
            drop_last=False
        )
    data, labels = get_space(None, dataloader, mode='data') 
    return data, labels

def save_all(data, latents, labels):
    np.savetxt("csv/data.csv", data, delimiter=",") 
    np.savetxt("csv/latents.csv", latents, delimiter=",")
    np.savetxt("csv/labels.csv", labels, delimiter=",")

def topo_magic(data, latents):
    pass

def main():
    latents, latents_labels  = load_latents() 
    data, labels = load_data()
    save_all(data, latents, labels)    

    topo_magic(data, latents)

if __name__ == '__main__': 
    main()




    



      

 
    
