"""Vanilla models."""
import torch.nn as nn

from src.models import submodules
from .base import AutoencoderModel


class ConvolutionalAutoencoderModel(submodules.ConvolutionalAutoencoder):
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


class VanillaAutoencoderModel(AutoencoderModel):
    def __init__(self, autoencoder_model='ConvolutionalAutoencoder',
                 ae_kwargs=None):
        super().__init__()
        ae_kwargs = ae_kwargs if ae_kwargs else {}
        self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)

    def forward(self, x):
        return self.autoencoder(x)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)
