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
        'k': 15,
        'evaluate_on': 'test'
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

    supports_transform = hasattr(model, 'transform')
    if supports_transform:
        # Models that support fitting on train and predicting on test
        data, labels = zip(*train_dataset)
    else:
        # Models which do not derive an mapping to the latent space
        _log.warn('Model does not support separate training and prediction.')
        _log.warn('Directly running on test dataset!')
        data, labels = zip(*test_dataset)

    data = np.array(data)
    labels = np.array(labels)

    _log.info('Fitting model...')
    transformed_data = model.fit_transform(data)

    rundir = None
    try:
        rundir = _run.observers[0].dir
    except IndexError:
        pass

    if rundir:
        # Save model state (and entire model)
        with open(os.path.join(rundir, 'model.pth'), 'wb') as f:
            pickle.dump(model, f)
        np.save(os.path.join(rundir, 'transformed_data.npy'),
                transformed_data)

    result = {}
    if evaluation['active']:
        evaluate_on = evaluation['evaluate_on']
        _log.info(f'Running evaluation on {evaluate_on} dataset')
        if supports_transform:
            if evaluate_on == 'validation':
                data, labels = zip(*validation_dataset)
                data = np.array(data)
                labels = np.array(labels)
                transformed_data = model.transform(data)
            else:
                # Load dedicated test dataset and predict on it
                data, labels = zip(*test_dataset)
                data = np.array(data)
                labels = np.array(labels)
                transformed_data = model.transform(data)

        evaluator = Multi_Evaluation(
            dataloader=None, seed=_seed, model=None)
        ev_result = evaluator.evaluate_space(
            data, transformed_data, labels, K=evaluation['k'])
        prefixed_ev_result = {
            evaluate_on + '_' + key: value
            for key, value in ev_result.items()
        }
        result.update(prefixed_ev_result)

    return result
