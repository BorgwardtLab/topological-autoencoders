"""Topolologically regularized autoencoder using approximation."""
import torch
import torch.nn as nn

from ..topology import PersistentHomologyCalculation
from .base import AutoencoderModel
from .submodules import ConvolutionalAutoencoder


class TopologicallyRegularizedAutoencoder(AutoencoderModel):
    """Topologically regularized autoencoder."""

    def __init__(self, lam=1.):
        """Topologically Regularized Autoencoder.

        Args:
            lam: Regularization strength
        """
        super().__init__()
        self.lam = lam
        self.topo_sig = TopologicalSignature()
        self.autoencoder = ConvolutionalAutoencoder()

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1) ** 0.5

    def forward(self, x):
        """Compute the loss of the Topologically regularized autoencoder.

        Args:
            x: Input data

        Returns:
            Tuple of final_loss, (...loss components...)

        """
        batch_size = x.size()[0]
        latent = self.autoencoder.encode(x)
        reconst = self.autoencoder.decode(latent)

        # We currently use convolutional autoencoders, thus it might be
        # necessary to flatten the input prior to the computation of
        # topological signatures.
        x_flat = x.view(batch_size, -1)
        latent_flat = latent.view(batch_size, -1)

        sig_data = self.topo_sig(x_flat, norm=True)
        sig_latent = self.topo_sig(latent_flat)

        # Use reconstruction loss of autoencoder
        reconst_error = self.autoencoder.reconst_error(x, reconst)
        topo_error = self.sig_error(sig_data, sig_latent)

        loss = reconst_error + self.lam * topo_error
        return (
            loss,
            {
                'reconst_error': reconst_error,
                'topo_error': topo_error
            }
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)


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

