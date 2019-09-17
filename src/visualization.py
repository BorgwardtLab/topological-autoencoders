"""Functions for visualizing stuff."""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_latents(latents, labels, save_file=None):
    plt.scatter(latents[:, 0], latents[:, 1], c=labels,
                cmap=plt.cm.Spectral, s=2., alpha=0.5)
    if save_file:
        plt.savefig(save_file, dpi=200)
        plt.close()


def plot_losses(losses, losses_std=defaultdict(lambda: None), save_file=None):
    """Plot a dictionary with per epoch losses.

    Args:
        losses: Mean of loss per epoch
        losses_std: stddev of loss per epoch

    """
    for key, values in losses.items():
        plt.errorbar(range(len(values)), values, yerr=losses_std[key], label=key)

    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    if save_file:
        plt.savefig(save_file, dpi=200)
        plt.close()


def shape_is_image(shape):
    """Check if is a 4D tensor which we consider to be an image."""
    return len(shape) == 4


def visualize_n_reconstructions_from_dataset(dataset, inverse_normalization,
                                             model, n_reconst, output_path):
    import torch
    from itertools import islice
    from torchvision.utils import save_image
    vis_data, _ = zip(*islice(dataset, None, n_reconst))
    vis_data = np.stack(vis_data)
    vis_latent = model.encode(torch.tensor(vis_data))
    reconst_images = model.decode(vis_latent)
    reconst_images = inverse_normalization(reconst_images)
    save_image(
        reconst_images,
        output_path
    )
