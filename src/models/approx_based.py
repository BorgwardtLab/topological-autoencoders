"""Topolologically regularized autoencoder using approximation."""
import torch
import torch.nn as nn

from src.topology import PersistentHomologyCalculation
from src.models import submodules
from src.models.base import AutoencoderModel


class TopologicallyRegularizedAutoencoder(AutoencoderModel):
    """Topologically regularized autoencoder."""

    def __init__(self, lam=1., autoencoder_model='ConvolutionalAutoencoder',
                 ae_kwargs=None, toposig_kwargs=None):
        """Topologically Regularized Autoencoder.

        Args:
            lam: Regularization strength
            ae_kwargs: Kewords to pass to `ConvolutionalAutoencoder` class
            toposig_kwargs: Keywords to pass to `TopologicalSignature` class
        """
        super().__init__()
        self.lam = lam
        ae_kwargs = ae_kwargs if ae_kwargs else {}
        toposig_kwargs = toposig_kwargs if toposig_kwargs else {}
        self.topo_sig = TopologicalSignature(**toposig_kwargs)
        self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)

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

    def __init__(self, p=2, sort_selected=False, use_cycles=False):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.p = p
        self.sort_selected = sort_selected
        self.use_cycles = use_cycles
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
        pairs_0, pairs_1 = self.signature_calculator(distances.detach().numpy())

        if self.sort_selected:
            # Sort arrays in-place
            pairs_0.sort()
            pairs_1.sort()

        selected_distances = distances[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            selected_cycle_distances = \
                distances[(pairs_1[:, 0], pairs_1[:, 1])]

            selected_distances = torch.cat(
                (selected_distances, selected_cycle_distances))

        return selected_distances

