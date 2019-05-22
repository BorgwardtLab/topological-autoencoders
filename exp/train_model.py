"""Module to train a model with a dataset configuration."""
import os

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
import torch
import numpy as np
import pandas as pd

from src.callbacks import Callback, SaveReconstructedImages, \
    SaveLatentRepresentation, Progressbar
from src.datasets.splitting import split_validation
from src.evaluation.eval import Multi_Evaluation
from src.evaluation.utils import get_space
from src.training import TrainingLoop
from src.visualization import plot_losses, visualize_latents

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
    val_size = 0.15
    early_stopping = 10
    device = 'cuda'
    quiet = False
    evaluation = {
        'active': False,
        'k_min': 10,
        'k_max': 200,
        'k_step': 10,
        'evaluate_on': 'test',
        'online_visualization': False,
        'save_latents': True
    }


@EXP.named_config
def rep1():
    seed = 249040430

@EXP.named_config
def rep2():
    seed = 621965744

@EXP.named_config
def rep3():
    seed=771860110

@EXP.named_config
def rep4():
    seed=775293950


class NewlineCallback(Callback):
    """Add newline between epochs for better readability."""
    def on_epoch_end(self, **kwargs):
        print()


@EXP.automain
def train(n_epochs, batch_size, learning_rate, weight_decay, val_size,
          early_stopping, device, quiet, evaluation, _run, _log, _seed, _rnd):
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
    model.to(device)

    callbacks = [
        LogTrainingLoss(_run, print_progress=quiet),
        LogDatasetLoss('validation', validation_dataset, _run,
                       print_progress=True, batch_size=batch_size,
                       early_stopping=early_stopping, device=device),
        LogDatasetLoss('testing', test_dataset, _run, print_progress=True,
                       batch_size=batch_size, device=device),
    ]

    if quiet:
        # Add newlines between epochs
        callbacks.append(NewlineCallback())
    else:
        callbacks.append(Progressbar(print_loss_components=True))

    # If we are logging this run save reconstruction images
    rundir = None
    try:
        rundir = _run.observers[0].dir
        if hasattr(dataset, 'inverse_normalization'):
            # We have image data so we can visualize reconstructed images
            callbacks.append(SaveReconstructedImages(rundir))
        if evaluation['online_visualization']:
            callbacks.append(
                SaveLatentRepresentation(
                    test_dataset, rundir, batch_size=64, device=device)
            )

    except IndexError:
        pass

    training_loop = TrainingLoop(
        model, dataset, n_epochs, batch_size, learning_rate, weight_decay,
        device, callbacks
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
    loss_averages = {
        key: value for key, value in logged_averages.items() if 'loss' in key
    }
    loss_stds = {
        key: value for key, value in logged_stds.items() if 'loss' in key
    }
    metric_averages = {
        key: value for key, value in logged_averages.items() if 'metric' in key
    }
    metric_stds = {
        key: value for key, value in logged_stds.items() if 'metric' in key
    }
    if rundir:
        plot_losses(
            loss_averages,
            loss_stds,
            save_file=os.path.join(rundir, 'loss.png')
        )
        plot_losses(
            metric_averages,
            metric_stds,
            save_file=os.path.join(rundir, 'metrics.png')
        )

    result = {
        key: values[-1] for key, values in logged_averages.items()
    }

    if evaluation['active']:
        evaluate_on = evaluation['evaluate_on']
        _log.info(f'Running evaluation on {evaluate_on} dataset')
        if evaluate_on == 'validation':
            selected_dataset = validation_dataset
        else:
            selected_dataset = test_dataset

        dataloader = torch.utils.data.DataLoader(
            selected_dataset, batch_size=batch_size, pin_memory=True,
            drop_last=True
        )
        data, labels = get_space(None, dataloader, mode='data', seed=_seed)
        latent, _ = get_space(model, dataloader, mode='latent', device=device,
                              seed=_seed)

        if rundir and evaluation['save_latents']:
            df = pd.DataFrame(latent)
            df['labels'] = labels
            df.to_csv(os.path.join(rundir, 'latents.csv'), index=False)
            np.savez(
                os.path.join(rundir, 'latents.npz'),
                latents=latent, labels=labels
            )

        if latent.shape[1] == 2 and rundir:
            # Visualize latent space
            visualize_latents(
                latent, labels,
                save_file=os.path.join(rundir, 'latent_visualization.pdf')
            )

        k_min, k_max, k_step = \
            evaluation['k_min'], evaluation['k_max'], evaluation['k_step']
        ks = list(range(k_min, k_max + k_step, k_step))

        evaluator = Multi_Evaluation(
            dataloader=dataloader, seed=_seed, model=model)
        ev_result = evaluator.get_multi_evals(
            data, latent, labels, ks=ks)
        prefixed_ev_result = {
            evaluation['evaluate_on'] + '_' + key: value
            for key, value in ev_result.items()
        }
        result.update(prefixed_ev_result)

    return result
