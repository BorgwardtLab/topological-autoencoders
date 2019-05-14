'''
Utility functions for computing kNN vote for latent space evaluations
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score


'''
Takes data matrix X and number of nearest neighbors k and returns distances and indices of the nearest neighbors for each data point in X
'''
def get_k_nb(X, k=2):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X) #'ball_tree'
    distances, indices = nbrs.kneighbors(X)
    return distances, indices

'''
Majority Voting Function (hard vote, RANDOM choice upon ties)
'''
def make_vote(labels):
    unique, counts = np.unique(labels, return_counts=True)
    max_count = max(counts) #we choose randomized vote for ties
    winners = np.where(counts == max_count)
    vote_ind = np.random.choice(winners[0], 1)
    vote = unique[vote_ind]
    return vote
    
'''
Function to retrieve NN label predictions by sequentially including up to k neighbors:
'''
def get_k_predictions(X, y, k=10):
    # Loop to get all k neighboring labels for k-resolved label votes:
    n_neighbors = k
    n_samples = X.shape[0]

    #determine neighborhood of each sample: #[ n_samples x n_neighbors ] 
    distances, indices = get_k_nb(X, n_neighbors)

    #get labels of neighborhood [ n_samples x n_neighbors-1]  --> we drop the original data point
    neighboring_labels = y[indices][:,1:] 

    #Loop for voting:
    predicted_labels = np.empty([n_samples, n_neighbors-1])
    for k in np.arange(n_neighbors-1):
        for i in np.arange(n_samples):
            labels = neighboring_labels[i,:k+1]
            y_pred = make_vote(labels) #predicts most frequent label in current neighborhood
            predicted_labels[i,k] = y_pred
    return predicted_labels

'''
Get normalized mutual information for all k predictions
'''
def get_NMI(k_predictions, y_true):
    n_neighbors = k_predictions.shape[1]
    k_NMI = np.zeros(n_neighbors)
    for k in np.arange(n_neighbors):
        k_NMI[k] = normalized_mutual_info_score(y_true, k_predictions[:,k], average_method='arithmetic')
    return np.array(k_NMI)

'''
Get accuracies for all k predictions
'''
def get_acc(k_predictions, y_true):
    n_neighbors = k_predictions.shape[1]
    k_acc = np.zeros(n_neighbors)
    for k in np.arange(n_neighbors):
        k_acc[k] = accuracy_score(y_true, k_predictions[:,k], normalize=True)
    return np.array(k_acc)



