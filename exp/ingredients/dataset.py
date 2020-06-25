"""Module containing sacred functions for handling ML models."""
import inspect

from sacred import Ingredient

from src import datasets

ingredient = Ingredient('dataset')


@ingredient.config
def cfg():
    """Dataset configuration."""
    name = ''
    parameters = {
    }


@ingredient.named_config
def MNIST():
    """MNIST dataset."""
    name = 'MNIST'
    parameters = {
    }


@ingredient.named_config
def FashionMNIST():
    """FashionMNIST dataset."""
    name = 'FashionMNIST'
    parameters = {
    }

@ingredient.named_config
def CIFAR():
    """CIFAR10 dataset."""
    name = 'CIFAR'
    parameters = {
    }

@ingredient.named_config
def Spheres():
    name ='Spheres'
    parameters = {
    }

@ingredient.capture
def get_instance(name, parameters, _log, **kwargs):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    # Capture arguments passed to get_instance and pass to constructor
    parameters.update(kwargs)
    # Get the mode class
    model_cls = getattr(datasets, name)

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
        if parameter_name not in parameters.keys():
            # If an optional parameter is not defined warn and run with default
            default = available_parameters[parameter_name].default
            _log.warning(f'Optional parameter {parameter_name} not explicitly '
                         f'defined, will run with {parameter_name}={default}')

    return model_cls(**parameters)
