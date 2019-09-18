"""Module to train a competitor with a dataset configuration."""
import os

import numpy as np
import pandas as pd
import pickle

from sacred import Experiment

from src.datasets.splitting import split_validation
from src.evaluation.eval import Multi_Evaluation
from src.visualization import visualize_latents, shape_is_image

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
        'k_min': 10,
        'k_max': 200,
        'k_step': 10,
        'evaluate_on': 'test',
        'save_latents': True,
        'save_model': False,
        'save_training_latents': False,
        'n_reconstructions': 32
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

@EXP.named_config
def rep5():
    seed=700134501



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
    supports_inverse_transform = hasattr(model, 'inverse_transform')

    data, labels = zip(*train_dataset)
    if not supports_transform:
        # Models which do not derive an mapping to the latent space
        _log.warn('Model does not support separate training and prediction.')
        _log.warn('Will run evaluation on subsample of training dataset!')

    data = np.stack(data).reshape(len(data), -1)
    train_labels = np.array(labels)

    _log.info('Fitting model...')
    transformed_data = model.fit_transform(data)

    rundir = None
    try:
        rundir = _run.observers[0].dir
    except IndexError:
        pass

    if rundir and evaluation['save_model']:
        # Save model state (and entire model)
        with open(os.path.join(rundir, 'model.pth'), 'wb') as f:
            pickle.dump(model, f)

    result = {}
    if evaluation['active']:
        evaluate_on = evaluation['evaluate_on']
        _log.info(f'Running evaluation on {evaluate_on} dataset')
        if supports_transform:
            if evaluate_on == 'validation':
                data, labels = zip(*validation_dataset)
            else:
                # Load dedicated test dataset and predict on it
                data, labels = zip(*test_dataset)
            data = np.stack(data)
            original_data_shape = data.shape
            data = data.reshape(len(data), -1)
            labels = np.array(labels)
            latent = model.transform(data)
            if supports_inverse_transform:
                reconstructions = model.inverse_transform(latent)
                mse = np.mean((data - reconstructions) ** 2)
                print(f'Reconstruction error on {evaluate_on}: {mse}')
                result[f'{evaluate_on}_mse'] = mse

                if rundir and shape_is_image(original_data_shape):
                    print('Saving reconsturction images')
                    import torch
                    from torchvision.utils import save_image
                    n_reconst = evaluation['n_reconstructions']
                    reconst_images = reconstructions[:n_reconst]
                    reconst_images = np.reshape(
                        reconst_images, (n_reconst,) + original_data_shape[1:])
                    reconst_images = torch.tensor(reconst_images)
                    save_image(
                        test_dataset.inverse_normalization(reconst_images),
                        os.path.join(rundir, 'reconstruction.png')
                    )
        else:
            # If the model does not support transforming after fitting, take
            # a subset of the training data to compute evaluation metrics and
            # store latents
            indices = _rnd.permutation(len(train_dataset))
            indices = indices[:len(test_dataset)]
            data = data[indices]
            latent = transformed_data[indices]
            labels = train_labels[indices]


        if rundir and evaluation['save_latents']:
            df = pd.DataFrame(latent)
            df['labels'] = labels
            df.to_csv(os.path.join(rundir, 'latents.csv'), index=False)
            np.savez(
                os.path.join(rundir, 'latents.npz'),
                latents=latent, labels=labels
            )

        if rundir and evaluation['save_training_latents']:
            if supports_transform:
                train_data, train_labels = zip(*dataset)
                train_data = np.stack(train_data).reshape(len(train_data), -1)
                train_labels = np.array(train_labels)
                train_latent = model.transform(train_data)
            else:
                train_latent = transformed_data

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

        k_min, k_max, k_step = \
            evaluation['k_min'], evaluation['k_max'], evaluation['k_step']
        ks = list(range(k_min, k_max + k_step, k_step))

        evaluator = Multi_Evaluation(
            dataloader=None, seed=_seed, model=None)
        ev_result = evaluator.get_multi_evals(
            data, latent, labels, ks=ks)
        prefixed_ev_result = {
            evaluate_on + '_' + key: value
            for key, value in ev_result.items()
        }
        result.update(prefixed_ev_result)

    return result
