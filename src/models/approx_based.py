"""Topolologically regularized autoencoder using approximation."""
import numpy as np
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
        self.topo_sig = TopologicalSignatureDistance(**toposig_kwargs)
        self.autoencoder = getattr(submodules, autoencoder_model)(**ae_kwargs)

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    def forward(self, x):
        """Compute the loss of the Topologically regularized autoencoder.

        Args:
            x: Input data

        Returns:
            Tuple of final_loss, (...loss components...)

        """
        batch_size = x.size()[0]
        latent = self.autoencoder.encode(x)

        x_distances = self._compute_distance_matrix(x)
        # TODO: Normalize the distances in the data space --> does this make
        # sense?
        x_distances = x_distances / x_distances.max()
        latent_distances = self._compute_distance_matrix(latent)

        # Use reconstruction loss of autoencoder
        ae_loss, ae_loss_comp = self.autoencoder(x)

        topo_error, topo_error_components = self.topo_sig(
            x_distances, latent_distances)

        loss = ae_loss + self.lam * topo_error
        loss_components = {
            'loss.autoencoder': ae_loss,
            'loss.topo_error': topo_error
        }
        loss_components.update(topo_error_components)
        loss_components.update(ae_loss_comp)
        return (
            loss,
            loss_components
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)


class AlephPersistenHomologyCalculation():
    def __init__(self, compute_cycles):
        import aleph
        self._aleph = aleph
        self.compute_cycles = compute_cycles

    def __call__(self, distance_matrix):
        if self.compute_cycles:
            pairs_0, pairs_1 = self._aleph.vietoris_rips_from_matrix_2d(
                distance_matrix)
            pairs_0 = np.array(pairs_0)
            pairs_1 = np.array(pairs_1)
        else:
            pairs_0 = self._aleph.vietoris_rips_from_matrix_1d(
                distance_matrix)
            pairs_0 = np.array(pairs_0)
            pairs_1 = None

        return pairs_0, pairs_1


class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.sort_selected = sort_selected
        self.use_cycles = use_cycles
        self.match_edges = match_edges

        self.signature_calculator = AlephPersistenHomologyCalculation(
            compute_cycles=use_cycles)
        # else:
        #     self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(distances.detach().numpy())

        if self.sort_selected:
            # Sort arrays in-place
            pairs_0.sort()
            pairs_1.sort()

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1) ** 0.5

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set((v1, v2) for v1, v2 in array)
        return float(len(to_set(pairs1[0]).intersection(to_set(pairs2[0]))))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)
        sig1 = self._select_distances_from_pairs(distances1, pairs1)
        sig2 = self._select_distances_from_pairs(distances2, pairs2)

        distance_components = {
            'metrics.matched_pairs': self._count_matching_pairs(pairs1, pairs2)
        }

        if self.match_edges is None:
            distance = self.sig_error(sig1, sig2)
        elif self.match_edges == 'mask-one-way':
            # Ensure that gradients are computed using the matching edge in the
            # data space. For this we create a sparse tensor and convert it
            # back to a dense one, s.th. all unselected edges are zero.
            pairs1_tensor = torch.LongTensor(pairs1[0])
            pairs1_distances = torch.sparse.FloatTensor(
                pairs1_tensor.t(), sig1,
                torch.Size([distances1.size(0), distances1.size(0)])).to_dense()
            # From this masked distance tensor, we pick the edges which
            # correspond to the selected latent space egdes as these are the
            # only ones that result in gradients.
            sig1 = pairs1_distances[
                (pairs1[0][:, 0], pairs1[0][:, 1])]
            distance = self.sig_error(sig1, sig2)
        elif self.match_edges == 'symmetric':
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)

            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components
