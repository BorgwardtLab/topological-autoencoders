"""Module to train a model with a dataset configuration."""
from sacred import Experiment
from src.callbacks import SaveReconstructedImages, Progressbar
from src.training import TrainingLoop

from .ingredients import model as model_config
from .ingredients import dataset as dataset_config

EXP = Experiment(
    'training',
    ingredients=[model_config.ingredient, dataset_config.ingredient]
)


@EXP.config
def cfg():
    n_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    print_progress = True


# pylint: disable=E1120
@EXP.automain
def train(n_epochs, batch_size, learning_rate, _run, _log):
    """Sacred wrapped function to run training of model."""
    # Get data
    dataset = dataset_config.get_instance()
    # Get model
    model = model_config.get_instance()

    callbacks = [
        Progressbar(print_loss_components=True),
    ]

    # If we are logging this run save reconstruction images
    try:
        rundir = _run.observers[0].dir
        callbacks.append(SaveReconstructedImages(rundir))
    except IndexError:
        pass

    training_loop = TrainingLoop(
        model, dataset, n_epochs, batch_size, learning_rate,
        callbacks
    )
    # Run training
    training_loop()

