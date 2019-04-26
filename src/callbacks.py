"""Callbacks for training loop."""
from tqdm import tqdm

class Callback():
    """Callback for training loop."""

    def on_epoch_begin(self, local_variables):
        """Call before an epoch begins."""

    def on_epoch_end(self, local_variables):
        """Call after an epoch is finished."""

    def on_batch_begin(self, local_variables):
        """Call before a batch is being processed."""

    def on_batch_end(self, local_variables):
        """Call after a batch has be processed."""


class Progressbar(Callback):
    def __init__(self):
        self.progressbar = None

    def on_epoch_begin(self, local_variables):
        n_batches = local_variables['n_batches']
        self.progressbar = tqdm(total=n_batches, unit='batches')

    def on_batch_end(self, local_variables):
        loss = local_variables['loss']
        self.progressbar.update(1)
        self.progressbar.set_description(f'Loss: {loss:3.3f}')

    def on_epoch_end(self, local_variables):
        self.progressbar.close()
        self.prohressbar = None
