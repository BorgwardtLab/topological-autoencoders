
import numpy as np
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.decomposition import PCA, KernelPCA
import torch

#custom code:
from .utils import get_space, rescaling 
from .knn_utils import get_k_predictions, get_NMI, get_acc

torch.manual_seed(42)

'''
Evaluation object:
    Inputs:
    - model: (pickle object using torch.save() )
    - dataloader: torch iterable dataloader object of pre-specified dataset and train/val/test split
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

#def plot_scores(ax, method, embeddings, true_labels, metric, k=40):
#    #first get predicted labels (based on up to k neighbors)
#    k_preds = get_k_predictions(embeddings[method], true_labels, k=k)
#    
#    #for each int up to k get a NMI score of how well the neighborhood labels vote for the correct label
#    scores = metric(k_preds, used_labels)
#    ax.plot( np.arange(1,len(scores)+1), scores, label=method)
#    
#    return 
#
#
## In[36]:
#
#
#
#methods = ['topological','vanilla', 'surrogate', 'original']
#
#tsne_embeddings = {
#            methods[0]: Z_tsne_top,
#            methods[1]: Z_tsne_van,
#            methods[2]: Z_tsne_surr,
#            methods[3]: X_tsne
#    
#}
#
#f, (ax1, ax2) = plt.subplots(1, 2)
#for method in methods:
#    plot_scores(ax1, method, tsne_embeddings, used_labels, metric=get_NMI, k=50)
#
#plt.ylim((0., 1)) 
#plt.ylabel('Normalized Mutual Information')
#plt.xlabel('Number of k Nearest Neighbors for Label Prediction')
#
#plt.legend()
#
#plt.title('Quality of Embedding after applying t-SNE on Latent (or Data) space')
#plt.savefig('nmi_scores.pdf')
#
#
## In[ ]:
#
#
##Neighborhood Scoring without tsne (no visualization! )
#
#methods = ['topological','vanilla', 'surrogate', 'original']
#
#embeddings = {
#            methods[0]: topo_latent,
#            methods[1]: vanilla_latent,
#            methods[2]: surr_latent,
#            methods[3]: data_scaled
#}    
#
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
#for method in methods:
#    plot_scores(ax1, method, embeddings, used_labels, metric=get_NMI, k=100)
#ax1.set_ylim((0., 1)) 
#ax1.set_ylabel('Normalized Mutual Information')
#ax1.set_xlabel('Number of k Nearest Neighbors for Label Prediction')
#ax1.legend()
#
#for method in methods:
#    plot_scores(ax2, method, embeddings, used_labels, metric=get_acc, k=100)
#ax2.set_ylim((0., 1)) 
#ax2.set_ylabel('Accuracy')
#ax2.set_xlabel('Number of k Nearest Neighbors for Label Prediction')
#ax2.legend()
#
##plt.legend()
#f.suptitle(f"Quantitative Evaluation of Latent (and Data) Spaces using {n_samples}", fontsize=14)
#
#plt.savefig('scores_native.pdf') #native latent spaces without additional tsne embedding for visualization
#
#
#
#

#HERE IS THE EMBEDDING AND PLOTTING CODE TO INCLUDE ABOVE:

#
#tsne = TSNE(random_state=42, perplexity=20) #, perplexity=10, learning_rate=100, early_exaggeration=30)
#Z_tsne_top = tsne.fit_transform(topo_latent)
#
#tsne = TSNE(random_state=42, perplexity=20) #, perplexity=10, learning_rate=100)
#Z_tsne_van = tsne.fit_transform(vanilla_latent)
#
#tsne = TSNE(random_state=42, perplexity=20) #, perplexity=10, learning_rate=100)
#X_tsne = tsne.fit_transform(data_scaled)
#
#
#tsne = TSNE(random_state=42, perplexity=20) #, perplexity=10, learning_rate=100, early_exaggeration=30)
#Z_tsne_surr = tsne.fit_transform(surr_latent)
#
#
#pca = PCA(random_state=0)
#Z_pca_top = pca.fit_transform(topo_latent)
#
#pca = PCA(random_state=0)
#Z_pca_van = pca.fit_transform(vanilla_latent)
#
#pca = PCA(random_state=0)
#Z_pca_surr = pca.fit_transform(surr_latent)
#
#pca = PCA(random_state=0)
#X_pca = pca.fit_transform(data_scaled)
#
#
#def plotting(transformed, all_labels, title):
#    color_mapping = matplotlib.cm.rainbow(np.linspace(0, 1, 10))
#    distinct_labels = np.unique(all_labels)
#    colors = [color_mapping[cl] for cl in distinct_labels]
#    for i, label in enumerate(distinct_labels):
#        mask = (all_labels == label).astype(int)
#        inds = list(np.where(mask)[0])
#        plt.scatter(transformed[inds, 0], transformed[inds, 1], c=[colors[i]], label=label, s=2)
#    plt.title(title)
#    lgnd = plt.legend(loc="lower left")
#    for i,_ in enumerate(distinct_labels):
#        lgnd.legendHandles[i]._sizes = [30]
#
#plotting(Z_tsne_top, all_labels[:n_samples], 't-SNE Latent Space Differentiable Topo-AE {n_samples} samples')
#
#plotting(Z_tsne_van, all_labels[:n_samples], 't-SNE Latent Space Vanilla-AE {n_samples} samples')
#
#plotting(Z_tsne_surr, all_labels[:n_samples], 't-SNE Latent Space Surrogate Topo-AE {n_samples} samples')
#
#plotting(X_tsne, all_labels[:n_samples], '2D Embedding of Data Space {n_samples} samples')
#
#plotting(Z_pca_top, all_labels[:n_samples], f'PCA Latent Space Differentiable Topo-AE {n_samples} samples')
#
#plotting(Z_pca_van, all_labels[:n_samples], f'PCA Latent Space Vanilla AE {n_samples} samples')
#
#plotting(Z_pca_surr, all_labels[:n_samples], f'PCA Latent Space Surrogate Topo-AE {n_samples} samples')
#
#plotting(X_pca, all_labels[:n_samples], f'PCA Data Space {n_samples} samples')
#

