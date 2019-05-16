"""Submodules used by models."""
import torch
import torch.nn as nn
from torch.distributions import Normal

from .base import AutoencoderModel
# Hush the linter: Warning W0221 corresponds to a mismatch between parent class
# method signature and the child class
# pylint: disable=W0221


class ConvolutionalAutoencoder(AutoencoderModel):
    """Convolutional Autoencoder for MNIST/Fashion MNIST."""

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
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class ConvolutionalAutoencoder_2D(AutoencoderModel):
    """Convolutional Autoencoder."""

    def __init__(self):
        """Convolutional Autoencoder."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 2, 3, stride=2, padding=1),  # b, 2, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=1)  # b, 2, 1, 1
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 16, 5),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class ConvolutionalAutoencoder_STL10(AutoencoderModel):
    """Convolutional Autoencoder Architecture for STL10."""

    def __init__(self):
        """Convolutional Autoencoder."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=3, padding=1),  # b, 16, 31, 31
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 15, 15
            nn.Conv2d(16, 8, 5, stride=2, padding=1),  # b, 8, 7, 7 
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=3)  # b, 8, 2, 2 
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 8, 6, stride=3, padding=1),  # b, 8, 46, 46
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 8, stride=2, padding=1),  # b, 3, 96, 96
            nn.Tanh()
        )
        self.reconst_error = nn.MSELoss()

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class MLPAutoencoder(AutoencoderModel):
    def __init__(self, arch=[3, 32, 32, 2]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 3)
        )
        self.reconst_error = nn.MSELoss()

    @staticmethod
    def _build_layers(arch, final_bias, final_relu):
        layers = []
        for i, (d_in, d_out) in enumerate(zip(arch, arch[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i == len(arch)-2 and not final_relu:
                layers.append(nn.ReLU(True))
        return layers

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}



class MLPAutoencoder_Spheres(AutoencoderModel):
    def __init__(self, arch=[3, 32, 32, 2]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(101, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 101)
        )
        self.reconst_error = nn.MSELoss()

    @staticmethod
    def _build_layers(arch, final_bias, final_relu):
        layers = []
        for i, (d_in, d_out) in enumerate(zip(arch, arch[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i == len(arch)-2 and not final_relu:
                layers.append(nn.ReLU(True))
        return layers

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        return self.encoder(x)

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        return self.decoder(z)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent = self.encode(x)
        x_reconst = self.decode(latent)
        reconst_error = self.reconst_error(x, x_reconst)
        return reconst_error, {'reconstruction_error': reconst_error}


class MLPVAE(AutoencoderModel):
    def __init__(self, input_dim=3, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim*2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, input_dim*2)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def _split_to_parameters(self, x):
        return x.split(x.size(-1) // 2, dim=-1)

    def encode(self, x):
        """Compute latent representation using convolutional autoencoder."""
        mu, logvar = self._encode_latent_parameters(x)
        encoded = self.reparameterize(mu, logvar)
        return encoded

    def decode(self, z):
        """Compute reconstruction using convolutional autoencoder."""
        mu, logvar = self._decode_latent_parameters(z)
        encoded = self.reparameterize(mu, logvar)
        return encoded

    def log_likelihood(self, x, reconst_mean, reconst_std):
        predicted_x = Normal(loc=reconst_mean, scale=reconst_std)
        return predicted_x.log_prob(x).sum(dim=-1)

    def forward(self, x):
        """Apply autoencoder to batch of input images.

        Args:
            x: Batch of images with shape [bs x channels x n_row x n_col]

        Returns:
            tuple(reconstruction_error, dict(other errors))

        """
        latent_mu, latent_logvar = self._split_to_parameters(self.encoder(x))
        latent = self.reparameterize(latent_mu, latent_logvar)

        data_mu, data_logvar = self._split_to_parameters(self.decoder(latent))
        data_std = torch.exp(0.5*data_logvar)

        likelihood = -self.log_likelihood(x, data_mu, data_std).mean(dim=0)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(
            1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp(),
            dim=-1
        ).mean(dim=0)
        loss = likelihood + kl_div
        return loss, {'likelihood': likelihood, 'kl_divergence': kl_div}
