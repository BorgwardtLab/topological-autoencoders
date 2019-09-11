"""Callbacks specific to sacred."""
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.callbacks import Callback


def convert_to_base_type(value):
    """Convert a value into a python base datatype.

    Args:
        value: numpy or torch value

    Returns:
        Python base type
    """
    if isinstance(value, (torch.Tensor, np.generic)):
        return value.item()
    else:
        return value


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
        loss = convert_to_base_type(loss)
        self.iterations += 1
        self.epoch_losses['training.loss'].append(loss)
        self.run.log_scalar('training.loss.batch', loss, self.iterations)
        for key, value in loss_components.items():
            value = convert_to_base_type(value)
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
                 batch_size=128, early_stopping=None, save_path=None,
                 device='cpu'):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            dataset_name: Name of dataset
            dataset: Dataset to use
            run: Sacred run
            print_progress: Print evaluated loss
            batch_size: Batch size
            early_stopping: if int the number of epochs to wait befor stopping
                training due to non-decreasing loss, if None dont use
                early_stopping
            save_path: Where to store model weigths
        """
        self.prefix = dataset_name
        self.dataset = dataset
        # TODO: Ideally drop last should be set to false, yet this is currently
        # incompatible with the surrogate approach as it assumes a constant
        # batch size.
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size,
                                      drop_last=True, pin_memory=True)
        self.run = run
        self.print_progress = print_progress
        self.early_stopping = early_stopping
        self.save_path = save_path
        self.device = device
        self.iterations = 0
        self.patience = 0
        self.best_loss = np.inf

    def _compute_average_losses(self, model):
        losses = defaultdict(list)
        model.eval()

        for batch in self.data_loader:
            data, _ = batch
            if self.device == 'cuda':
                data = data.cuda(non_blocking=True)
            loss, loss_components = model(data)
            loss = convert_to_base_type(loss)

            # Rescale the losses as batch_size might not divide dataset
            # perfectly, this currently is a nop as drop_last is set to True in
            # the constructor.
            n_instances = len(data)
            losses['loss'].append(loss*n_instances)
            for loss_component, value in loss_components.items():
                value = convert_to_base_type(value)
                losses[loss_component].append(
                    value*n_instances)
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
        if self.early_stopping is not None:
            if losses['loss'] < self.best_loss:
                self.best_loss = losses['loss']
                if self.save_path is not None:
                    save_path = os.path.join(self.save_path, 'model_state.pth')
                    print('Saving model to', save_path)
                    torch.save(
                        model.state_dict(),
                        save_path
                    )
                self.patience = 0
            else:
                self.patience += 1

            if self.early_stopping <= self.patience:
                print(
                    'Stopping training due to non-decreasing '
                    f'{self.prefix} loss over {self.early_stopping} epochs'
                )
                return True

