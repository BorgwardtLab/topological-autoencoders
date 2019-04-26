"""Callbacks for training loop."""
import os
from tqdm import tqdm
from torchvision.utils import save_image

class Callback():
    """Callback for training loop."""

    def on_epoch_begin(self, **local_variables):
        """Call before an epoch begins."""

    def on_epoch_end(self, **local_variables):
        """Call after an epoch is finished."""

    def on_batch_begin(self, **local_variables):
        """Call before a batch is being processed."""

    def on_batch_end(self, **local_variables):
        """Call after a batch has be processed."""


class Progressbar(Callback):
    def __init__(self, print_loss_components=False):
        self.print_loss_components = print_loss_components
        self.total_progress = None
        self.epoch_progress = None

    def on_epoch_begin(self, n_epochs, n_instances, **kwargs):
        if self.total_progress is None:
            self.total_progress = tqdm(
                position=0, total=n_epochs, unit='epochs')
        self.epoch_progress = tqdm(
            position=1, total=n_instances, unit='instances')

    def on_batch_end(self, batch_size, loss, loss_components, **kwargs):
        self.epoch_progress.update(batch_size)
        description = f'Loss: {loss:3.3f}'
        if self.print_loss_components:
            description += ', '
            description += ', '.join([
                f'{name}: {value:3.3f}'
                for name, value in loss_components.items()
            ])
        self.epoch_progress.set_description(description)

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        self.epoch_progress.close()
        self.epoch_progress = None
        self.total_progress.update(1)
        if epoch == n_epochs:
            self.total_progress.close()


class SaveReconstructedImages(Callback):
    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, model, dataset, img, epoch, **kwargs):
        latent = model.encode(img)
        reconst = model.decode(latent)
        reconstructed_image = dataset.inverse_normalization(reconst)
        save_image(
            reconstructed_image, os.path.join(self.path,f'epoch_{epoch}.png'))

