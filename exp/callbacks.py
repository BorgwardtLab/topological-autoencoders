"""Callbacks specific to sacred."""
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from src.callbacks import Callback


class LogTrainingLoss(Callback):
    """Logging of loss during training into sacred run."""

    def __init__(self, run, print_progress=False):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            run: Sacred run
        """
        self.run = run
        self.print_progress = print_progress
        self.epoch_losses = None
        self.logged_averages = defaultdict(list)
        self.logged_stds = defaultdict(list)
        self.iterations = 0

    def _description(self):
        all_keys = self.logged_averages.keys()
        elements = []
        for key in all_keys:
            last_average = self.logged_averages[key][-1]
            last_std = self.logged_stds[key][-1]
            elements.append(
                f'{key}: {last_average:3.3f} +/- {last_std:3.3f}')
        return ' '.join(elements)

    def on_epoch_begin(self, **kwargs):
        self.epoch_losses = defaultdict(list)

    def on_batch_end(self, loss, loss_components, **kwargs):
        self.iterations += 1
        loss = loss.cpu().item()
        self.epoch_losses['training.loss'].append(loss)
        self.run.log_scalar('training.loss.batch', loss, self.iterations)
        for key, value in loss_components.items():
            value = value.cpu().item()
            storage_key = 'training.' + key
            self.epoch_losses[storage_key].append(value)
            self.run.log_scalar(storage_key + '.batch', value, self.iterations)

    def on_epoch_end(self, epoch, **kwargs):
        for key, values in self.epoch_losses.items():
            mean = np.mean(values)
            std = np.std(values)
            self.run.log_scalar(key + '.mean', mean, self.iterations)
            self.logged_averages[key].append(mean)
            self.run.log_scalar(key + '.std', std, self.iterations)
            self.logged_stds[key].append(std)
        self.epoch_losses = defaultdict(list)
        if self.print_progress:
            print(f'Epoch {epoch}:', self._description())


class LogDatasetLoss(Callback):
    """Logging of loss during training into sacred run."""

    def __init__(self, dataset_name, dataset, run, print_progress=True,
                 batch_size=128):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            run: Sacred run
        """
        self.prefix = dataset_name
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size,
                                      drop_last=False)
        self.run = run
        self.print_progress = print_progress
        self.iterations = 0

    def _compute_average_losses(self, model):
        losses = defaultdict(list)

        for batch in self.data_loader:
            data, _ = batch
            loss, loss_components = model(data)

            # Rescale the losses as batch_size might not divide dataset
            # perfectly
            n_instances = len(data)
            losses['loss'].append(loss.item()*n_instances)
            for loss_component, value in loss_components.items():
                losses[loss_component].append(
                    value.item()*n_instances)
        return {
            name: sum(values) / len(self.dataset)
            for name, values in losses.items()
        }

    def _progress_string(self, epoch, losses):
        progress_str = " ".join([
            f'{self.prefix}.{key}: {value:.3f}'
            for key, value in losses.items()
        ])
        return f'Epoch {epoch}: ' + progress_str

    def on_batch_end(self, **kwargs):
        self.iterations += 1

    def on_epoch_begin(self, model, epoch, **kwargs):
        """Store the loss on the dataset prior to training."""
        if epoch == 1:  # This should be prior to the first training step
            losses = self._compute_average_losses(model)
            if self.print_progress:
                print(self._progress_string(epoch - 1, losses))

            for key, value in losses.items():
                self.run.log_scalar(
                    f'{self.prefix}.{key}',
                    value,
                    self.iterations
                )

    def on_epoch_end(self, model, epoch, **kwargs):
        """Score evaluation metrics at end of epoch."""
        losses = self._compute_average_losses(model)
        print(self._progress_string(epoch, losses))
        for key, value in losses.items():
            self.run.log_scalar(
                f'{self.prefix}.{key}',
                value,
                self.iterations
            )
