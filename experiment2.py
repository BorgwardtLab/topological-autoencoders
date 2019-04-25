"""Experiment 2, can we implement persistence reg. without surrogate."""
import os

from src.models import TopologicallyRegularizedAutoencoder
from src.datasets import MNIST
from src.training import TrainingLoop

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def main():
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3


    model = TopologicallyRegularizedAutoencoder()
    dataset = MNIST()
    training_loop = TrainingLoop(
        model, dataset, num_epochs, batch_size, learning_rate)
    training_loop()


    torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == '__main__':
    main()
