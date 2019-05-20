"""Module to train a competitor with a dataset configuration."""
import os

import numpy as np
import pickle

from sacred import Experiment

from src.datasets.splitting import split_validation
from src.evaluation.eval import Multi_Evaluation

from .ingredients import model as model_config
from .ingredients import dataset as dataset_config

EXP = Experiment(
    'fit_competitor',
    ingredients=[model_config.ingredient, dataset_config.ingredient]
)


@EXP.config
def config():
    val_size = 0.15
    evaluation = {
        'active': False,
        'k': 15
    }


@EXP.automain
def train(val_size, evaluation, _run, _log, _seed, _rnd):
    """Sacred wrapped function to run training of model."""
    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123

    dataset = dataset_config.get_instance(train=True)
    train_dataset, validation_dataset = split_validation(
        dataset, val_size, _rnd)
    test_dataset = dataset_config.get_instance(train=False)

    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120
    model = model_config.get_instance()

    train_data_flattened = np.array(
        [X for X, y in train_dataset]
    )

    transformed_train = model.fit_transform(train_data_flattened)

    rundir = None
    try:
        rundir = _run.observers[0].dir
    except IndexError:
        pass

    if rundir:
        # Save model state (and entire model)
        with open(os.path.join(rundir, 'model.pth'), 'wb') as f:
            pickle.dump(model, f)
        np.save(os.path.join(rundir, 'train_transformed.npy'),
                transformed_train)

    result = {}
    if evaluation['active']:
        data = train_data_flattened
        labels = np.array(
            [y for X, y in train_dataset]
        )
        latent = transformed_train

        evaluator = Multi_Evaluation(
            dataloader=None, seed=_seed, model=None)
        ev_result = evaluator.evaluate_space(
            data, latent, labels, K=evaluation['k'])
        result.update(ev_result)

    return result
