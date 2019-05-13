'''
Utility Functions for evaluation scenario
'''
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

'''
get_latent_space: 
    This function takes a model and dataloader object, 
    feeds all samples through the model and extracts and concatenates 
    samples from the required space and the labels
    --> tested only on image data (mnist, ..)
    - mode: 'latent' --> extracts latent codes
            'data'   --> extracts data samples
'''

def get_space(model, dataloader, mode='latent', seed=42):
    #initialize output space    
    full_space = []
    all_labels = []
    
    torch.manual_seed(seed)

    if mode == 'data':
        #Extract data samples 
        for index, batch in enumerate(dataloader):
            #unpack current batch:
            image, label = batch

            im = image.detach().numpy()
            im_flat = im.reshape(im.shape[0], -1)
            full_space.append(im_flat)
            all_labels.append(label)

        #Concatenate the lists to return arrays of data space and labels
        full_space = np.concatenate(full_space, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    
    elif mode == 'latent':
        #Extract latent codes from latent space    
        for index, batch in enumerate(dataloader):
            #unpack current batch:
            image, label = batch

            #feed batch through model:
            latent = model.encode(image)
            reconst = model.decode(latent)
            
            #extract latent code and flatten to vector
            latent = latent.detach().numpy()
            latent_flat = latent.reshape(latent.shape[0], -1)
            
            #extract reconstructed image 
            reconst = reconst.detach().numpy()
            
            #append current latent code and label to list of all 
            full_space.append(latent_flat)
            all_labels.append(label)

        #Concatenate the lists to return arrays of latent codes and labels
        full_space = np.concatenate(full_space, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        raise ValueError(f'Requested mode: {mode} not available. Instead use one of: \'latent\', \'data\' ')       
    
    return [full_space, all_labels]



'''
Function to scale a dataset (e.g. data space, latent space) before applying embeddings for visualization
'''
def rescaling(data):
    return StandardScaler().fit_transform(data)



