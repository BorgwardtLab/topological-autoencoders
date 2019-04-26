"""Topologically constrained autoencoder using learned surrogate."""
import aleph
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AutoencoderModel
from .submodules import ConvolutionalAutoencoder

# Hush the linter: Warning W0221 corresponds to a mismatch between parent class
# method signature and the child class
# pylint: disable=W0221


class TopologicalSurrogateAutoencoder(AutoencoderModel):
    """Topologically constrained autoencoder using learned surrogate."""

    def __init__(self, d_latent, batch_size, arch, lam1=1., lam2=1., eps=1e9,
                 dim=1):
        """Topologically constrained autoencoder using learned surrogate.

        Args:
            d_latent: Dimensionality of latent space
            batch_size: Batch size
            arch: Architecture for topological signature estimation
            lam1: Regularize difference in topological signatures between x and
                z
            lam2: Regularize exactness of signature approximation
            eps: Maximally expected distance
            dim: Dimensionality of topological signatures to compute
        """
        super().__init__()
        self.autoencoder = ConvolutionalAutoencoder()
        self.sig_comp = SignatureComputation(eps, dim)
        self.sig_estim = SignatureEstimator(
            d_latent, batch_size, arch)
        self.reconst_error = nn.MSELoss()
        self.lam1 = lam1
        self.lam2 = lam2

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
        latent = self.autoencoder.encode(x)
        reconstructed = self.autoencoder.decode(latent)

        pers_x = self.sig_comp(x)
        pers_z = self.sig_comp(latent)

        estim_pers_z = self.sig_estim(latent)

        reconst_error = self.reconst_error(x, reconstructed)
        topo_error = self.sig_error(pers_x, estim_pers_z)
        surrogate_error = self.sig_error(pers_z, estim_pers_z)

        loss = \
            reconst_error \
            + self.lam1 * topo_error \
            + self.lam1 * surrogate_error

        return (
            loss,
            {
                'reconstruction_error': reconst_error,
                'topological_error': topo_error,
                'surrogate_error': surrogate_error
            }
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)


class SignatureComputation(nn.Module):
    """Compute topological signatures using aleph."""

    def __init__(self, eps, dim):
        """Compute topological signatures using aleph.

        Args:
            eps: Maximum of filtration
            dim: Maximal dimensionality of topological features
        """
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x, norm=False):
        """Compute topological signatures using aleph.

        Args:
            x: Data
            norm: Normalize data output using maximal distance of MNIST
        """
        batch_size = x.size(0)
        x_detached = x.view(batch_size, -1).detach().numpy().astype(np.float64)
        pers_x = aleph.calculatePersistenceDiagrams(
            x_detached, self.eps, self.dim)[0]
        pers_x = np.array(pers_x)[:, 1]
        if norm:
            # Divide by maximal distance on MNIST
            pers_x /= 39.5
        pers_x[~np.isfinite(pers_x)] = 0
        pers_x = torch.tensor(pers_x, dtype=torch.float)
        return pers_x


class SignatureEstimator(nn.Module):
    """Neural network for the estimation of persistence signatures."""

    def __init__(self, d_in, batch_size, arch):
        """Neural netwoek for the estimation of persistence signatures.

        Args:
            d_in: Dimensionality of a single instance
            batch_size: Size of a batch
            arch: List of hidden layer sizes
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(a, b)
            for a, b in zip(
                [d_in * batch_size] + arch, arch + [batch_size])
        ])

    def forward(self, x):
        """Estimate topological signature of batch."""
        batch_size = x.size(0)
        # Flatten input
        out = x.view(-1)
        for layer in self.layers:
            out = F.relu(layer(out))
        return out.view(batch_size)
