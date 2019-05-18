'''
Trying out Tadasets library for generating topological synthetic datasets: 
Here an overlay of a torus, sphere and swiss roll
'''

import numpy as np 
#import tadasets 
import matplotlib
import matplotlib.pyplot as plt
from .custom_shapes import dsphere 

from IPython import embed

def create_sphere_dataset(n_samples=500, d=100, n_spheres=11, r=5, plot=False, seed=42):
    np.random.seed(seed)

    #it seemed that rescaling the shift variance by sqrt of d lets big sphere stay around the inner spheres
    variance=10/np.sqrt(d)

    shift_matrix = np.random.normal(0,variance,[n_spheres, d+1])

    spheres = [] 
    n_datapoints = 0
    for i in np.arange(n_spheres-1):
        sphere = dsphere(n=n_samples, d=d, r=r)
        spheres.append(sphere + shift_matrix[i,:])
        n_datapoints += n_samples

    #Additional big surrounding sphere:
    n_samples_big = 10*n_samples #int(n_samples/2)
    big = dsphere(n=n_samples_big, d=d, r=r*5)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color])
        plt.show()

    #Create Dataset:
    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints) 
    label_index=0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index:label_index + n_sphere_samples] = index
        label_index += n_sphere_samples
    
    return dataset, labels 
