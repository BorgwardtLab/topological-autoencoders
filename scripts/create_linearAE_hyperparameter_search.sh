#!/bin/bash

## SPHERES

ae_models=(TopoRegEdgeSymmetric)
output_pattern='experiments/hyperparameter_search/dimensionality_reduction/{dataset}/LinearAE-{model}.json'

# Autoencoder based
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${ae_models[*]} \
  --name dataset  \
  --set Spheres \
  --name dummy --set overrides.model__parameters__autoencoder_model=LinearAE_Spheres \
  --output-pattern ${output_pattern}



## REAL WORLD

output_pattern='experiments/hyperparameter_search/real_world/{dataset}/LinearAE-{model}.json'
input_dims='[3,32,32]'

#AE methods:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${ae_models[*]} \
  --name dataset --set MNIST FashionMNIST \
  --name dummy --set overrides.model__parameters__autoencoder_model=LinearAE \
  --output-pattern ${output_pattern}

python scripts/configs_from_product.py exp.hyperparameter_search \
    --name model \
    --set ${ae_models[*]} \
    --name dataset --set CIFAR \
    --name dummy --set overrides.model__parameters__autoencoder_model=LinearAE \
    --name dummy --set overrides.model__parameters__ae_kwargs__input_dims=${input_dims} \
    --output-pattern ${output_pattern}

