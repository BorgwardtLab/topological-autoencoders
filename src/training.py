"""Training classes."""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


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


class TrainingLoop():
    def __init__(self, model, dataset, n_epochs, batch_size, learning_rate,
                 callbacks=None):
        self.model = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.callbacks = callbacks if callbacks else []

    def _execute_callbacks(self, hook, local_variables):
        for callback in self.callbacks:
            getattr(callback, hook)(**local_variables)

    def on_epoch_begin(self, **local_variables):
        """Call callbacks before an epoch begins."""
        self._execute_callbacks('on_epoch_begin', local_variables)

    def on_epoch_end(self, **local_variables):
        """Call callbacks after an epoch is finished."""
        self._execute_callbacks('on_epoch_end', local_variables)

    def on_batch_begin(self, **local_variables):
        """Call callbacks before a batch is being processed."""
        self._execute_callbacks('on_batch_begin', local_variables)

    def on_batch_end(self, **local_variables):
        """Call callbacks after a batch has be processed."""
        self._execute_callbacks('on_batch_end', local_variables)

    def __call__(self):
        """Execute the training loop."""
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        for epoch in range(self.n_epochs):
            for i, data in enumerate(dataloader):
                img, _ = data
                img = Variable(img)  #.cuda()

                # Autoencoder
                loss, (reconst_error, topo_error) = self.model(img)

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
                  .format(epoch+1, self.n_epochs, loss.data.item()))


