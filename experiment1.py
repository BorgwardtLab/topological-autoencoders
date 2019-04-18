import aleph
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class PersistenceEstimator(nn.Module):
    def __init__(self, d_in, batch_size, arch):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(a, b)
            for a, b in zip(
                [d_in * batch_size] + arch, arch + [batch_size])
        ])

    def forward(self, x):
        bs, _ = x.size()
        # Flatten input
        out = x.view(-1)
        for layer in self.layers[:-1]:
            out = F.relu(layer(out))
        return self.layers[-1](out).view(bs, 1)


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
        bs, n_row, n_col = x.size()
        latent = self.encoder(x)
        x_reconst = self.decoder(latent)
        return latent.view(bs, -1), x_reconst


class TopologicalRegularization(nn.Module):
    def __init__(self, d_latent, batch_size, arch, eps=1e9, dim=1):
        self.persistence_estimator = PersistenceEstimator(
            d_latent, batch_size, arch)
        self.eps = eps
        self.dim = dim

    def _pers_dist(self, p1, p2):
        return (p1 - p2)**2

    def forward(self, x, z):
        pers_x = aleph.calculatePersistenceDiagrams(
            x.numpy(), self.eps, self.dim)[0]
        pers_x = np.array(pers_x)[:, 1]
        pers_x[~np.isfinite(pers_x)] = 100

        pers_z = aleph.calculatePersistenceDiagrams(
            z.numpy(), self.eps, self.dim)[0]
        pers_z = np.array(pers_z)[:, 1]
        pers_z[~np.isfinite(pers_z)] = 100

        approx_pers_z = self.persistence_estimator(z)

        return (
            self._pers_dist(pers_x, approx_pers_z),
            self._pers_dist(pers_z, approx_pers_z)
        )


def main():
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3
    lam1 = 1.
    lam2 = 1.

    img_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder() #.cuda()
    reg = TopologicalRegularization(8*2*2, batch_size, [256, 256])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = Variable(img)  #.cuda()

            latent, reconstructed = model(img)
            reconst_error = criterion(reconstructed, img)
            topo_reg, topo_approx = reg(img, latent)
            loss = reconst_error + lam1*topo_reg + lam2*topo_approx

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.data.item() )) #loss.data[0] 
        if epoch % 1 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == '__main__':
    main()
