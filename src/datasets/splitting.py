"""Image Dataset Splitting (train val)."""
from math import floor
import numpy as np
from torch.utils.data import Dataset, Subset, SubsetRandomSampler
from torch._utils import _accumulate
import torch


def split_validation(dataset, val_fraction, _rnd):
    """Randomly split a dataset into two non-overlapping new datasets.

    Arguments:
        dataset (Dataset): Dataset to be split
        val_fraction: Fraction of dataset that should be in the validation
            split
        _rnd: Random state to determine split
    """
    assert val_fraction < 1.
    indices = _rnd.permutation(len(dataset))
    last_train_index = int(floor((1 - val_fraction) * len(dataset)))
    return (
        Subset(dataset, indices[:last_train_index]),  # train split
        Subset(dataset, indices[last_train_index:])   # validation split
    )


'''
Torch dataloader splitting function:

inputs: dataset object, validation_size (determines ratio of val and test split)
'''

def split_dataset(dataset, val_size=0.2, batch_size=64):
    """Torch dataloader splitting function.

    Args:
        dataset: Dataset to split
        val_size: Fraction of data
        batch_size:

    Returns:
    """
    # Creating data indices for training, validation and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_size * dataset_size))
    np.random.shuffle(indices) #should take sacred internal seed
    train_indices, val_indices, test_indices = indices[2*split:], indices[split:2*split], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    split_loaders = []
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    split_loaders.append(train_loader)
    
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    split_loaders.append(validation_loader) 
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=test_sampler)
    split_loaders.append(test_loader)
    
    return split_loaders


#
##First version, with sklearn API:
#
#from manifolds import normalize_features
#from sklearn.model_selection import train_test_split
#
#
##Some util functions to flatten and restoring image shapes (currently not in use anymore)
#def flatten(X): 
#    '''Flatten images'''
#    img_shape = list(X.shape[1:])
#    n_samples = X.shape[0]
#    X_flat = X.view(n_samples, -1) 
#    return X_flat, img_shape
#
#def to_img(X, img_shape): 
#    '''Reshaping to original image shape'''
#    n_samples = X.shape[0]
#    new_shape = [n_samples] + img_shape
#    X_img = X.view(new_shape)
#    return X_img 
#
##Class for splitting a dataset using sklearn API:     
#class Split_Dataset:
#    '''
#    This class takes a torchvision image dataset flattens the images (keeping the first dim constant - assuming it indicates the samples), splits the dataset into train / test and reshapes the images into the original shape.
#    '''
#    def __init__(self, X, y, test_fraction, random_seed): 
#        #convert to np as this func doesnt take torch tensors
#        X = np.array(X); y = np.array(y)
#        #split dataset into train / test ( or train / val since we apply it on train dataset)
#        X_train, X_test, y_train, y_test = train_test_split(
#            X, y, test_size=test_fraction, random_state=random_seed)
#        #return to original torch tensor
#        X_train = torch.from_numpy(X_train); X_test = torch.from_numpy(X_test)
#        y_train = torch.from_numpy(y_train); y_test = torch.from_numpy(y_test)
#        self.X_train, self.X_test = X_train, X_test
#        self.y_train, self.y_test = y_train, y_test
#
#                   
