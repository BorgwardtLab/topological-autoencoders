"""Base class for autoencoder models."""
import abc
from typing import Dict, Tuple

import torch.nn as nn


class AutoencoderModel(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for autoencoders."""

    # pylint: disable=W0221
    @abc.abstractmethod
    def forward(self, x) -> Tuple[float, Dict[str, float]]:
        """Compute loss for model.

        Args:
            x: Tensor with data

        Returns:
            Tuple[loss, dict(loss_component_name -> loss_component)]

        """

    @abc.abstractmethod
    def encode(self, x):
        """Compute latent representation."""

    @abc.abstractmethod
    def decode(self, z):
        """Compute reconstruction."""

