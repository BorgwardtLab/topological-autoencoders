"""Image Dataset Splitting (train val)."""
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from manifolds import normalize_features
import torch

#Some util functions to flatten and restoring image shapes (currently not in use anymore)
def flatten(X): 
    '''Flatten images'''
    img_shape = list(X.shape[1:])
    n_samples = X.shape[0]
    X_flat = X.view(n_samples, -1) 
    return X_flat, img_shape

def to_img(X, img_shape): 
    '''Reshaping to original image shape'''
    n_samples = X.shape[0]
    new_shape = [n_samples] + img_shape
    X_img = X.view(new_shape)
    return X_img 
     

class Split_Dataset:
    '''
    This class takes a torchvision image dataset flattens the images (keeping the first dim constant - assuming it indicates the samples), splits the dataset into train / test and reshapes the images into the original shape.
    '''
    def __init__(self, X, y, test_fraction, random_seed): 
        #convert to np as this func doesnt take torch tensors
        X = np.array(X); y = np.array(y)
        #split dataset into train / test ( or train / val since we apply it on train dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=random_seed)
        #return to original torch tensor
        X_train = torch.from_numpy(X_train); X_test = torch.from_numpy(X_test)
        y_train = torch.from_numpy(y_train); y_test = torch.from_numpy(y_test)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

                   
