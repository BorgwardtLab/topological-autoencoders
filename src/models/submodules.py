"""Submodules used by models."""
import torch
from torch import nn

from ..topology import PersistentHomologyCalculation


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional Autoencoder."""

    def __init__(self):
        """Convolutional Autoencoder."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    # pylint: disable=W0221
    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            flattened_latent_representation, reconstructed images

        """
        batch_size = x.size()[0]
        latent = self.encoder(x)
        x_reconst = self.decoder(latent)
        return latent.view(batch_size, -1), x_reconst


class TopologicalSignature(nn.Module):
    """Topological signature."""

    def __init__(self, p=2):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
        """
        super().__init__()
        self.p = p
        self.signature_calculator = PersistentHomologyCalculation()

    # pylint: disable=W0221
    def forward(self, x, norm=False):
        """Take a batch of instances and return the topological signature.

        Args:
            x: batch of instances
            norm: Normalize computed distances by maximum value
        """
        distances = torch.norm(x[:, None] - x, dim=2, p=self.p)
        if norm:
            distances = distances / distances.max()
        pairs = self.signature_calculator(distances.detach().numpy())
        selected_distances = distances[(pairs[:, 0], pairs[:, 1])]
        return selected_distances
