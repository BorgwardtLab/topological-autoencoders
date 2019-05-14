"""Module with scared experiment for hyperparameter search."""
import os
from tempfile import NamedTemporaryFile
from collections import defaultdict
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import set_by_dotted_path

import skopt

from .train_model import EXP as train_experiment

ex = Experiment('hyperparameter_search')
ex.observers.append(FileStorageObserver.create('hyperparameter_search_runs'))


@ex.config
def cfg():
    """Hyperparameter search configuration."""
    hyperparameter_space = {
        # Training parameters
        'learning_rate': ('Real', 10**-4, 10**-2, 'log-uniform')
    }
    overrides = {
        'batch_size': 64,
        'evaluation__active': True,
        'n_epochs': 100,
        'quiet': True
    }
    nan_replacement = 20.
    n_random_starts = 10
    n_calls = 30
    load_result = None  # load the previous optimization results from here


@ex.named_config
def TopoRegMNIST():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.1, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
        'dataset__name': 'MNIST'
    }


@ex.named_config
def TopoRegVertexMNIST():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.1, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
        'model__parameters__toposig_kwargs__sort_selected': True,
        'dataset__name': 'MNIST'
    }


@ex.named_config
def TopoRegVertexFashionMNIST():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.1, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
        'model__parameters__toposig_kwargs__sort_selected': True,
        'dataset__name': 'FashionMNIST'
    }


@ex.named_config
def TopoRegEdgeMNIST():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.1, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
        'model__parameters__match_distances': True,
        'dataset__name': 'MNIST'
    }


@ex.named_config
def TopoRegEdgeFashionMNIST():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.1, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
        'model__parameters__match_distances': True,
        'dataset__name': 'FashionMNIST'
    }


@ex.named_config
def VanillaMNIST():
    overrides = {
        'model__name': 'VanillaAutoencoderModel',
        'dataset__name': 'FashionMNIST'

    }


@ex.named_config
def VanillaFashionMNIST():
    overrides = {
        'model__name': 'VanillaAutoencoderModel',
        'dataset__name': 'FashionMNIST'
    }


@ex.capture
def build_search_space(hyperparameter_space):
    """Build scikit-optimize hyperparameter search space from configuration."""
    space = []
    for name, subspace in hyperparameter_space.items():
        parameter_type = subspace[0]
        range_definition = subspace[1:]
        if hasattr(skopt.space, parameter_type):
            parameter_class = getattr(skopt.space, parameter_type)
            space.append(parameter_class(*range_definition, name=name))
        else:
            ValueError(f'{parameter_type} is not a valid parameter_type')
    return space


def remove_functions_from_skopt_res(res):
    """Remove function from skopt result as they are not pickleable."""
    # Make copy of result and remove references that would make it
    # hard to work with later (in particular the objective function
    # and the callback stored in the result
    res = deepcopy(res)
    res_without_func = deepcopy(res)
    del res_without_func.specs['args']['func']
    del res_without_func.specs['args']['callback']
    return res_without_func


class SkoptCheckpointCallback:
    """Save intermediate results of hyperparameter search."""

    def __init__(self, filename):
        """Create callback."""
        self.filename = filename

    def __call__(self, res):
        """Store skop result into file."""
        res_without_func = remove_functions_from_skopt_res(res)
        skopt.dump(res_without_func, self.filename)


def prepare_evaluation_results(parameters, results):
    """Convert parameter and results log into dictionary of lists."""
    combined_dict = defaultdict(list)
    for parameters, results in zip(parameters, results):
        for parameter, value in parameters.items():
            combined_dict[parameter].append(value)
        for result, value in results.items():
            combined_dict[result].append(value)
    return dict(combined_dict)


@ex.main
def search_hyperparameter_space(n_random_starts, n_calls, overrides,
                                nan_replacement, load_result, _rnd, _run,
                                _log):
    """Search hyperparameter space of an experiment."""
    # Add observer to child experiment to store all intermediate results
    if _run.observers:
        run_dir = _run.observers[0].dir
        train_experiment.observers.append(
            FileStorageObserver.create(os.path.join(run_dir, 'model_runs')))

        # Also setup callback to store intermediate hyperparameter search
        # results in a checkpoint file
        callbacks = [
            SkoptCheckpointCallback(
                os.path.join(run_dir, 'result_checkpoint.pck'))
        ]
    else:
        callbacks = []

    # Setup search space
    search_space = build_search_space()

    # Setup objective and logging of all run results
    results = []
    evaluated_parameters = []
    _run.result = {}

    @skopt.utils.use_named_args(search_space)
    def objective(**params):
        for key in params.keys():
            if isinstance(params[key], np.int64):
                # Strangeley, it seems like we dont get 'real' ints here,
                # but a numpy datatypes. So lets cast them.
                params[key] = int(params[key])
        # Need to do this here in order to get rid
        # of leftovers from previous evaluations
        plt.close('all')

        # Update the parameters we go with constant overrides
        params.update(overrides)

        # Transform the search space and overrides into structure of nested
        # dicts
        # This workaround as sacred does not allow '.' in dict keys
        params = {
            key.replace('__', '.'): value for key, value in params.items()
        }
        evaluated_parameters.append(params)

        # Convert to nested dict
        transformed_params = {}
        for key, value in params.items():
            # This function is from sacred and used to convert x.y=z notation
            # into nested dicts: {'x': {'y': z}}
            set_by_dotted_path(transformed_params, key, value)

        _log.debug(f'Running training with parameters: {transformed_params}')
        try:
            # Run the experiment and update config according to overrides
            # to overrides and sampled parameters
            run = train_experiment.run(config_updates=transformed_params)
            results.append(run.result)

            # gp optimize does not handle nan values, thus we need
            # to return something fake if we diverge before the end
            # of the first epoch
            if np.isfinite(run.result['nmis_avg']):
                return_value = -run.result['nmis_avg']
            else:
                return_value = nan_replacement
        except Exception as e:
            _log.error('An exception occured during fitting: {}'.format(e))
            results.append({})
            return_value = nan_replacement

        # Store the results into sacred infrastructure
        # Ensures they can be used even if the experiment is terminated
        parameter_evaluations = prepare_evaluation_results(
            evaluated_parameters, results)
        _run.result['parameter_evaluations'] = parameter_evaluations
        with NamedTemporaryFile(suffix='.csv') as f:
            df = pd.DataFrame.from_dict(parameter_evaluations)
            df.to_csv(f.name)
            _run.add_artifact(f.name, 'parameter_evaluations.csv')
        return return_value

    # Load previous evaluations if given
    if load_result:
        _log.info('Loading previous evaluations from {}'.format(load_result))
        with _run.open_resource(load_result, 'rb') as f:
            loaded_res = skopt.load(f)
        x0 = loaded_res.x_iters
        y0 = loaded_res.func_vals
        if n_random_starts != 0:
            _log.warning('n_random_starts is {} and not 0, '
                         'will evaluate further random points '
                         'after loading stored result'.format(n_random_starts))
    else:
        x0 = None
        y0 = None

    res_gp = skopt.gp_minimize(
        objective, search_space, x0=x0, y0=y0, n_calls=n_calls,
        n_random_starts=n_random_starts, random_state=_rnd, callback=callbacks
    )

    # Store final optimization results
    with NamedTemporaryFile(suffix='.pck') as f:
        res_without_func = remove_functions_from_skopt_res(res_gp)
        skopt.dump(res_without_func, f.name)
        _run.add_artifact(f.name, 'result.pck')

    parameter_evaluations = prepare_evaluation_results(
        evaluated_parameters, results)
    best_parameters = {
        variable.name: value
        for variable, value in zip(search_space, res_gp.x)
    }
    return {
        'parameter_evaluations': parameter_evaluations,
        'Best score': res_gp.fun,
        'best_parameters': best_parameters
    }


if __name__ == '__main__':
    ex.run_commandline()
