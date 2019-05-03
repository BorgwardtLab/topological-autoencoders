"""Module to train a model with a dataset configuration."""
import os

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import torch

from src.callbacks import SaveReconstructedImages, Progressbar
from src.training import TrainingLoop
from src.visualization import plot_losses

from .callbacks import LogTrainingLoss
from .ingredients import model as model_config
from .ingredients import dataset as dataset_config

EXP = Experiment(
    'training',
    ingredients=[model_config.ingredient, dataset_config.ingredient]
)
EXP.captured_out_filter = apply_backspaces_and_linefeeds


@EXP.config
def cfg():
    n_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    weight_decay = 1e-5


@EXP.automain
def train(n_epochs, batch_size, learning_rate, weight_decay, _run, _log,
          _seed):
    """Sacred wrapped function to run training of model."""
    torch.manual_seed(_seed)
    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    dataset = dataset_config.get_instance(train=True)

    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120
    model = model_config.get_instance()

    callbacks = [
        LogTrainingLoss(_run),
        Progressbar(print_loss_components=True)
    ]

    # If we are logging this run save reconstruction images
    rundir = None
    try:
        rundir = _run.observers[0].dir
        if hasattr(dataset, 'inverse_normalization'):
            callbacks.append(SaveReconstructedImages(rundir))
    except IndexError:
        pass

    training_loop = TrainingLoop(
        model, dataset, n_epochs, batch_size, learning_rate, weight_decay,
        callbacks
    )
    # Run training
    training_loop()


    logged_averages = callbacks[0].logged_averages
    logged_stds = callbacks[0].logged_stds
    if rundir:
        plot_losses(
            logged_averages,
            logged_stds,
            save_file=os.path.join(rundir, 'loss.png')
        )

    # Convert the default dict into a regular one
    return dict(logged_averages.items())


