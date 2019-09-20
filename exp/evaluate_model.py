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
from src.evaluation.utils import compute_reconstruction_error, get_space
from src.training import TrainingLoop
from src.visualization import plot_losses, visualize_latents, shape_is_image, \
    visualize_n_reconstructions_from_dataset

from .callbacks import LogDatasetLoss, LogTrainingLoss
from .ingredients import model as model_config
from .ingredients import dataset as dataset_config

EXP = Experiment(
    'evaluation',
    ingredients=[model_config.ingredient, dataset_config.ingredient]
)
EXP.captured_out_filter = apply_backspaces_and_linefeeds


@EXP.config
def cfg():
    # Keep these here so we can reuse the config for train model
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
        'online_visualization': True,
        'save_latents': True,
        'save_training_latents': False,
        'n_reconstructions': 32
    }
    state_dict_path = ''

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

@EXP.named_config
def rep5():
    seed=700134501




@EXP.automain
def evaluate(batch_size, val_size, device, evaluation, state_dict_path, _run,
             _log, _seed, _rnd):
    """Sacred wrapped function to run training of model."""
    assert state_dict_path != ''
    torch.manual_seed(_seed)
    rundir = None
    try:
        rundir = _run.observers[0].dir
    except IndexError:
        pass

    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    dataset = dataset_config.get_instance(train=True)
    train_dataset, validation_dataset = split_validation(
        dataset, val_size, _rnd)
    test_dataset = dataset_config.get_instance(train=False)

    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120
    model = model_config.get_instance()
    print(f'Loading model from {state_dict_path}...')
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    result = {}

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

        if rundir and evaluation['save_training_latents']:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, pin_memory=True,
                drop_last=True
            )
            train_latent, train_labels = get_space(
                model, dataloader, mode='latent', device=device, seed=_seed)

            df = pd.DataFrame(train_latent)
            df['labels'] = train_labels
            df.to_csv(os.path.join(rundir, 'train_latents.csv'), index=False)
            np.savez(
                os.path.join(rundir, 'latents.npz'),
                latents=train_latent, labels=train_labels
            )
            # Visualize latent space
            visualize_latents(
                train_latent, train_labels,
                save_file=os.path.join(
                    rundir, 'train_latent_visualization.pdf')
            )

        if latent.shape[1] == 2 and rundir:
            # Visualize latent space
            visualize_latents(
                latent, labels,
                save_file=os.path.join(rundir, 'latent_visualization.pdf')
            )

        if rundir and shape_is_image((1, ) + test_dataset[0][0].shape):
            output_path = os.path.join(rundir, 'reconstruction.png')
            visualize_n_reconstructions_from_dataset(
                selected_dataset,
                dataset.inverse_normalization,
                model,
                evaluation['n_reconstructions'],
                output_path
            )

        # Compute reconstruction error
        reconstruction_error = compute_reconstruction_error(
            selected_dataset,
            batch_size, model,
            device)
        print(f'Reconstruction error on {evaluate_on}: {reconstruction_error}')
        result[f'{evaluate_on}_mse'] = reconstruction_error

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
