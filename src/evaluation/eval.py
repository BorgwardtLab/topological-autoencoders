
import numpy as np
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA, KernelPCA
import torch

#custom code:
from .utils import get_space, rescaling 
from .knn_utils import get_k_predictions, get_NMI, get_acc

import measures

torch.manual_seed(42)

def evaluate_space(data, labels, k):
    """Evaluate Space with kNN-based classification (accuracy) and label
    clustering (NMI).

    - data: samples from data or latent space as a data matrix
    - labels: corresponding labels to the samples
    - k: number of neighbors up to which the evaluation is iterating.
    """
    k_preds = get_k_predictions(data, labels, k)
    nmis = get_NMI(k_preds, labels)
    accs = get_acc(k_preds, labels)
    result = {'nmis_avg': nmis.mean(),
              'nmis': nmis,
              'accs_avg': accs.mean(),
              'accs': accs }
    return result

'''
Evaluation object:
    Inputs:
    - model: (pickle object using torch.save() )
    - dataloader: torch iterable dataloader object of pre-specified dataset and train/val/test split
'''
class Multi_Evaluation:
    def __init__(self, dataloader, seed=42, model=None):
        self.method = method
        self.dataloader = dataloader
        self.seed = seed
        if model:
            self.model = model
    
    def get_data(self, mode):
        """Extract specified space (data or latent space).
        Inputs: mode ('data' or 'latent') to extract data or latent space 
        Takes dataloader and model and mode (data or latent space) and returns
            [data, labels].
        """
        return get_space(self.model, self.dataloader, mode, self.seed)

    def rescale(self, data):
        """Apply a sklearn standard scaler data matrix before dim red method."""
        return rescaling(data)

    def subsample(self, data, labels, n_samples):
        """Subsampling n_samples from data_matrix and labels."""
        return data[:n_samples,:], labels[:n_samples]

    def evaluate_space(self, data, latent, labels, K):
        """Evaluate Space with multiple evaluation metrics for NLDR.

        - data: samples from data as a data matrix
        - latent: samples from the latent space as a matrix
        - labels: corresponding labels of the samples
        """
        results = get_multi_evals(data, latent, labels, K)
        return results

    def get_multi_evals(data, latent, labels, K):
        '''
        Performs multiple evaluations for nonlinear dimensionality
        reduction.

        - data: data samples as matrix
        - latent: latent samples as matrix
        - labels: labels of samples
        '''

        X = data
        Z = latent

        stress = measures.stress(X, Z)
        rmse = measures.RMSE(X, Z)

        trustworthiness = np.array(
            [measures.trustworthiness(X, Z, k) for k in range(1, K+1)]
        )

        continuity = np.array(
            [measures.continuity(X, Z, k) for k in range(1, K+1)]
        )

        neighbourhood_loss = np.array(
            [measures.neighbouhood_loss(X, Z, k) for k in range(1, K+1)]
        )

        mrre = np.array(
            [measures.MRRE(X, Z, k) for k in range(1, K+1)]
        )

        result = {
            # Scalars
            'stress': stress,
            'rmse': rmse,
            # Arrays/lists
            'trustworthiness': trustworthiness.tolist(),
            'continuity': continuity.tolist(),
            'neighbourhood_loss': neighbourhood_loss.tolist(),
            'mrre': mrre.tolist(),
            # Means
            'mean_trustworthiness': trustworthiness.mean(),
            'mean_continuity': continuity.mean(),
            'mean_neighbourhood_loss': neighbourhood_loss.mean(),
            'mean_mrre': mrre.mean()
        }

        return result


'''
Evaluation object using KNN properties alone
    Inputs:
    - model: (pickle object using torch.save() )
    - dataloader: torch iterable dataloader object of pre-specified dataset and train/val/test split
    - method: indicating if working on the original data or latent samples 

'''

class Evaluation:
    def __init__(self, method, dataloader, n_samples=500, seed=42, model=None):
        self.method = method
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.seed = seed
        if model:
            self.model = model
        if method == 'original': #determine whether original data or latent space should be evaluated
            self.mode = 'data'
        else: 
            self.mode = 'latent' 

    def get_data(self):
        """Extract specified space (data or latent space).

        Takes dataloader and model and mode (data or latent space) and returns
            [data, labels].
        """
        return get_space(self.model, self.dataloader, self.mode, self.seed)

    def rescale(self, data):
        """Apply a sklearn standard scaler data matrix before dim red method."""
        return rescaling(data)

    def subsample(self, data, labels):
        """Subsampling n_samples from data_matrix and labels."""
        n_samples = self.n_samples
        return data[:n_samples,:], labels[:n_samples]

    def get_embedding(self, data, emb_method):
        """Embed data matrix using one of the available dim red methods: pca, tsne, .."""
        pass

    def plot_embedding(self, transformed):
        """Plot 2d embedding."""
        pass

    def evaluate_space(self, data, labels, k):
        """Evaluate Space with kNN-based classification (accuracy) and label
        clustering (NMI).

        - data: samples from data or latent space as a data matrix
        - labels: corresponding labels to the samples
        - k: number of neighbors up to which the evaluation is iterating.
        """
        k_preds = get_k_predictions(data, labels, k)
        nmis = get_NMI(k_preds, labels)
        accs = get_acc(k_preds, labels)
        result = {'nmis_avg': nmis.mean(),
                  'nmis': nmis,
                  'accs_avg': accs.mean(),
                  'accs': accs }
        return result

