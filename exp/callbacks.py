"""Callbacks specific to sacred."""
from collections import defaultdict

import numpy as np

from src.callbacks import Callback


class LogTrainingLoss(Callback):
    """Logging of loss during training into sacred run."""

    def __init__(self, run):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            run: Sacred run
        """
        self.run = run
        self.epoch_losses = None
        self.logged_averages = defaultdict(list)
        self.logged_stds = defaultdict(list)
        self.iterations = 0

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

    def on_epoch_end(self, **kwargs):
        for key, values in self.epoch_losses.items():
            mean = np.mean(values)
            std = np.std(values)
            self.run.log_scalar(key + '.mean', mean, self.iterations)
            self.logged_averages[key].append(mean)
            self.run.log_scalar(key + '.std', std, self.iterations)
            self.logged_stds[key].append(std)
        self.epoch_losses = defaultdict(list)
