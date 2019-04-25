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


class Autoencoder(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        bs, n_channels, n_row, n_col = x.size()
        latent = self.encoder(x)
        x_reconst = self.decoder(latent)
        return latent.view(bs, -1), x_reconst


class TopologicalSignature(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
        self.signature_calculator = PersistentHomologyCalculation()

    def forward(self, x):
        """Take a batch of instances and return the topological signature."""
        distances = torch.norm(x[:, None] - x, dim=2, p=self.p)
        pairs = self.signature_calculator(distances.detach().numpy())
        selected_distances = distances[(pairs[:, 0], pairs[:, 1])]
        return selected_distances


class TopologicalRegularization(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def _pers_dist(self, p1, p2):
        return ((p1 - p2)**2).sum(dim=-1) ** 0.5

    def forward(self, x1, x2):
        """Compute distance between two topological signatures.

        Args:
            x1:
            x2:

        Returns:
            Scalar
        """
        return self._pers_dist(x1, x2)


def main():
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    lam1 = 1.
    lam2 = 1.

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder() #.cuda()
    topo_sig = TopologicalSignature()
    reg = TopologicalRegularization()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        chain(model.parameters(), reg.parameters()),
        lr=learning_rate, weight_decay=1e-5
    )

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            img, _ = data
            img = Variable(img)  #.cuda()
            bs = img.size()[0]

            # Autoencoder
            latent, reconstructed = model(img)
            reconst_error = criterion(reconstructed, img)

            # Topological signatures
            topo_sig_data = topo_sig(img.view(bs, -1))
            topo_sig_latent = topo_sig(latent.view(bs, -1))

            # Regularization
            topo_reg = reg(topo_sig_data, topo_sig_latent)
            loss = reconst_error + lam1*topo_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    f'MSE: {reconst_error}, '
                    f'topo_reg: {topo_reg}'
                )
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data.item() )) #loss.data[0] 
        if epoch % 1 == 0:
            pic = to_img(reconstructed.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == '__main__':
    main()
