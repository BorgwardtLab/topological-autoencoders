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
    load_model = False
    evaluation = {
        'active': False,
        'k_min': 10,
        'k_max': 200,
        'k_step': 10,
        'evaluate_on': 'test',
        'online_visualization': False,
        'save_latents': True,
        'save_training_latents': True
    }


@EXP.automain
def evaluate(load_model, device, batch_size, quiet, val_size, evaluation, _run, _log,
             _seed, _rnd):
    torch.manual_seed(_seed)
    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    dataset = dataset_config.get_instance(train=True)
    train_dataset, validation_dataset = split_validation(
        dataset, val_size, _rnd)
    test_dataset = dataset_config.get_instance(train=False)

    model = torch.load(load_model)

    rundir = None
    try:
        rundir = _run.observers[0].dir
    except IndexError:
        pass

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

