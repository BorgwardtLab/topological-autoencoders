"""Functions for visualizing stuff."""
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

