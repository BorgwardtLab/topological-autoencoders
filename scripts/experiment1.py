import os

from src.models import TopologicalSurrogateAutoencoder
from src.datasets import MNIST
from src.training import TrainingLoop
from src.callbacks import Progressbar

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def main():
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3

    model = TopologicalSurrogateAutoencoder(
        8*2*2, batch_size, [256, 256, 256, 256])
    dataset = MNIST()
    training_loop = TrainingLoop(
        model, dataset, num_epochs, batch_size, learning_rate,
        [Progressbar(print_loss_components=True)]
    )
    training_loop()

    torch.save(model.state_dict(), './experiment1.pth')


if __name__ == '__main__':
    main()
