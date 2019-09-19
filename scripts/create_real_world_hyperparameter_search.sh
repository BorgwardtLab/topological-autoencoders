#!/bin/bash

ae_models=(Vanilla TopoRegEdgeSymmetric)
competitor_methods=(PCA TSNE Isomap UMAP)
output_pattern='experiments/hyperparameter_search/real_world/{dataset}/{model}.json'
input_dims='[3,32,32]'

output_pattern_vae='experiments/hyperparameter_search/real_world/{dataset}/VAE-{model}.json'

#AE methods:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${ae_models[*]} \
  --name dataset --set MNIST FashionMNIST \
  --name dummy --set overrides.model__parameters__autoencoder_model=DeepAE \
  --output-pattern ${output_pattern}

python scripts/configs_from_product.py exp.hyperparameter_search \
    --name model \
    --set ${ae_models[*]} \
    --name dataset --set CIFAR \
    --name dummy --set overrides.model__parameters__autoencoder_model=DeepAE \
    --name dummy --set overrides.model__parameters__ae_kwargs__input_dims=${input_dims} \
    --output-pattern ${output_pattern}

#VAE method:
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set Vanilla \
  --name dataset --set MNIST FashionMNIST \
  --name dummy --set overrides.model__parameters__autoencoder_model=DeepVAE \
  --output-pattern ${output_pattern_vae}

python scripts/configs_from_product.py exp.hyperparameter_search \
    --name model \
    --set Vanilla \
    --name dataset --set CIFAR \
    --name dummy --set overrides.model__parameters__autoencoder_model=DeepVAE \
    --name dummy --set overrides.model__parameters__ae_kwargs__input_dims=${input_dims} \
    --output-pattern ${output_pattern_vae}

#Classic, non-deep Baselines: 
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${competitor_methods[*]} \
  --name dataset --set MNIST FashionMNIST CIFAR \
  --output-pattern ${output_pattern}
