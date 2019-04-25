"""Models."""
import torch.nn as nn

from .submodules import ConvolutionalAutoencoder, TopologicalSignature


class TopologicallyRegularizedAutoencoder(nn.Module):
    """Topologically regularized autoencoder."""

    def __init__(self, lam=1.):
        """Topologically Regularized Autoencoder.

        Args:
            lam: Regularization strength
        """
        super().__init__()
        self.lam = lam
        self.autoencoder = ConvolutionalAutoencoder()
        self.topo_sig = TopologicalSignature()
        self.reconst_error = nn.MSELoss()

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1) ** 0.5

    # pylint: disable=W0221
    def forward(self, x):
        """Compute the loss of the Topologically regularized autoencoder.

        Args:
            x: Input data

        Returns:
            Tuple of final_loss, (...loss components...)

        """
        batch_size = x.size()[0]
        latent, reconst = self.autoencoder(x)
        sig_data = self.topo_sig(x.view(batch_size, -1), norm=True)
        sig_latent = self.topo_sig(latent)

        reconst_error = self.reconst_error(x, reconst)
        topo_error = self.sig_error(sig_data, sig_latent)
        return (
            reconst_error + self.lam * topo_error,
            (reconst_error, topo_error)
        )
