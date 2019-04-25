"""Experiment 2, can we implement persistence reg. without surrogate."""
from itertools import chain

import aleph
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

from topology import PersistentHomologyCalculation

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


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


def signature_distance(signature1, signature2):
    """Compute distance between two topological signatures."""
    return ((signature1 - signature2)**2).sum(dim=-1) ** 0.5


class TopoRegAutoencoder(nn.Module):
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
        self.sig_error = signature_distance
        self.reconst_error = nn.MSELoss()

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


def main():
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TopoRegAutoencoder()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            img, _ = data
            img = Variable(img)  #.cuda()

            # Autoencoder
            loss, (reconst_error, topo_error) = model(img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    f'MSE: {reconst_error}, '
                    f'topo_reg: {topo_error}'
                )
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data.item() )) #loss.data[0] 
        if epoch % 1 == 0:
            _, reconstructed = model.autoencoder(img)
            pic = to_img(reconstructed.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == '__main__':
    main()
