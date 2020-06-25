"""Module containing sacred functions for handling ML models."""
import inspect

from sacred import Ingredient

from src import models

ingredient = Ingredient('model')


@ingredient.config
def cfg():
    """Model configuration."""
    name = ''
    parameters = {
    }


@ingredient.named_config
def TopologicalSurrogateAutoencoder():
    """TopologicalSurrogateAutoencoder."""
    name = 'TopologicalSurrogateAutoencoder'
    parameters = {
        'd_latent': 8*2*2,
        'batch_size': 32,
        'arch': [256, 256, 256, 256]
    }


@ingredient.named_config
def Vanilla():
    name =  'VanillaAutoencoderModel'


@ingredient.named_config
def VAE():
    name =  'VanillaAutoencoderModel'
    parameters = {
        'autoencoder_model': 'MLPVAE'
    }


@ingredient.named_config
def TopoReg():
    name = 'TopologicallyRegularizedAutoencoder'
    parameters = {
        'toposig_kwargs': {
            'sort_selected': False
        }
    }

@ingredient.named_config
def TopoRegSorted():
    name = 'TopologicallyRegularizedAutoencoder'
    parameters = {
        'toposig_kwargs': {
            'sort_selected': True
        }
    }


@ingredient.named_config
def TopoRegEdgeSymmetric():
    name = 'TopologicallyRegularizedAutoencoder'
    parameters = {
        'toposig_kwargs': {
            'match_edges': 'symmetric'
        }
    }


@ingredient.named_config
def TopoRegEdgeRandom():
    name = 'TopologicallyRegularizedAutoencoder'
    parameters = {
        'toposig_kwargs': {
            'match_edges': 'random'
        }
    }

@ingredient.capture
def get_instance(name, parameters, _log, _seed):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    # Get the mode class
    model_cls = getattr(models, name)

    # Inspect if the constructor specification fits with additional_parameters
    signature = inspect.signature(model_cls)
    available_parameters = signature.parameters
    for key in parameters.keys():
        if key not in available_parameters.keys():
            # If a parameter is defined which does not fit to the constructor
            # raise an error
            raise ValueError(
                f'{key} is not available in {name}\'s Constructor'
            )

    # Now check if optional parameters of the constructor are not defined
    optional_parameters = list(available_parameters.keys())[4:]
    for parameter_name in optional_parameters:
        # Copy list beforehand, so we can manipulate the parameter dict in the
        # loop
        parameter_keys = list(parameters.keys())
        if parameter_name not in parameter_keys:
            if parameter_name != 'random_state':
                # If an optional parameter is not defined warn and run with
                # default
                default = available_parameters[parameter_name].default
                _log.warning(
                    f'Optional parameter {parameter_name} not explicitly '
                    f'defined, will run with {parameter_name}={default}'
                )
            else:
                _log.info(
                    f'Passing seed of experiment to model parameter '
                    '`random_state`.'
                )
                parameters['random_state'] = _seed

    return model_cls(**parameters)
