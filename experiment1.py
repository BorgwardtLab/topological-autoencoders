import os

from src.models import TopologicalSurrogateAutoencoder
from src.datasets import MNIST
from src.training import TrainingLoop

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def main():
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    lam1 = 1.
    lam2 = 1.

    model = TopologicalSurrogateAutoencoder(
        8*2*2, batch_size, [256, 256, 256, 256])
    dataset = MNIST()
    training_loop = TrainingLoop(
        model, dataset, num_epochs, batch_size, learning_rate)
    training_loop()

    #         if i % 10 == 0:
    #             print(
    #                 f'MSE: {reconst_error}, '
    #                 f'topo_reg: {topo_reg}, topo_approx: {topo_approx}'
    #             )
    #     # ===================log========================
    #     print('epoch [{}/{}], loss:{:.4f}'
    #           .format(epoch+1, num_epochs, loss.data.item() )) #loss.data[0] 
    #     if epoch % 1 == 0:
    #         pic = to_img(reconstructed.cpu().data)
    #         save_image(pic, './dc_img/image_{}.png'.format(epoch))

    # torch.save(model.state_dict(), './conv_autoencoder.pth')


if __name__ == '__main__':
    main()
