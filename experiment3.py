"""Experiment 2, can we implement persistence reg. without surrogate."""
import os

import torch

from src.callbacks import SaveReconstructedImages, Progressbar
from src.datasets import MNIST
from src.models import TopologicallyRegularizedAutoencoder
from src.training import TrainingLoop


def main():
    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')

    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    lam=10

    model = TopologicallyRegularizedAutoencoder(lam, two_dim=True)
    dataset = MNIST()
    callbacks = [
        Progressbar(print_loss_components=True),
        SaveReconstructedImages('./dc_img')
    ]

    training_loop = TrainingLoop(
        model, dataset, num_epochs, batch_size, learning_rate,
        callbacks
    )
    training_loop()

    torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == '__main__':
    main()
