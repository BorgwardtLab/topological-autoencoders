"""Module to train a model with a dataset configuration."""
import os

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import torch

from src.callbacks import SaveReconstructedImages, Progressbar
from src.datasets.splitting import split_validation
from src.evaluation.eval import Evaluation
from src.training import TrainingLoop
from src.visualization import plot_losses

from .callbacks import LogDatasetLoss, LogTrainingLoss
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
    val_size = 0.2
    quiet = False
    evaluation = {
        'active': False,
        'k': 15
    }


@EXP.automain
def train(n_epochs, batch_size, learning_rate, weight_decay, val_size,
          quiet, evaluation, _run, _log, _seed, _rnd):
    """Sacred wrapped function to run training of model."""
    torch.manual_seed(_seed)
    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    dataset = dataset_config.get_instance(train=True)
    train_dataset, validation_dataset = split_validation(
        dataset, val_size, _rnd)
    test_dataset = dataset_config.get_instance(train=False)

    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120
    model = model_config.get_instance()

    callbacks = [
        LogTrainingLoss(_run, print_progress=quiet),
        LogDatasetLoss('validation', validation_dataset, _run,
                       print_progress=True, batch_size=batch_size),
        LogDatasetLoss('testing', test_dataset, _run, print_progress=True,
                       batch_size=batch_size),
    ]
    if not quiet:
        callbacks.append(Progressbar(print_loss_components=True))

    # If we are logging this run save reconstruction images
    rundir = None
    try:
        rundir = _run.observers[0].dir
        if hasattr(dataset, 'inverse_normalization'):
            # We have image data so we can visualize reconstructed images
            callbacks.append(SaveReconstructedImages(rundir))
    except IndexError:
        pass

    training_loop = TrainingLoop(
        model, dataset, n_epochs, batch_size, learning_rate, weight_decay,
        callbacks
    )
    # Run training
    training_loop()

    if rundir:
        # Save model state (and entire model)
        torch.save(
            model.state_dict(), os.path.join(rundir, 'model_state.pth'))
        torch.save(model, os.path.join(rundir, 'model.pth'))
    logged_averages = callbacks[0].logged_averages
    logged_stds = callbacks[0].logged_stds
    if rundir:
        plot_losses(
            logged_averages,
            logged_stds,
            save_file=os.path.join(rundir, 'loss.png')
        )

    result = dict(logged_averages.items())

    if evaluation['active']:
        dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, drop_last=True)
        evaluator = Evaluation('latent', dataloader, model=model)
        data, latent = evaluator.get_data()
        ev_result = evaluator.evaluate_space(data, latent, k=evaluation['k'])
        result['evaluation'] = ev_result

    return result


