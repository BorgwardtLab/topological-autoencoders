"""Vanilla models."""
import torch.nn as nn

from .submodules import ConvolutionalAutoencoder


class ConvolutionalAutoencoderModel(ConvolutionalAutoencoder):
    """Convolutional autoencoder model.

    Same as the submodule but returns MSE loss.
    """

    def __init__(self):
        """Convolutional Autoencoder."""
        super().__init__()
        self.reconst_error = nn.MSELoss()

    def forward(self, x):
        """Return MSE reconstruction loss of convolutional autoencoder."""
        _, reconst = super().forward(x)
        return self.reconst_error(x, reconst), tuple()
