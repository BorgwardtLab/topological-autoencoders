#!/bin/bash

ae_models=(Vanilla TopoRegEdgeSymmetric)
ae_models_cycles=(model.TopoReg model.TopoRegSorted model.TopoRegEdgeSymmetric)
competitor_methods=(PCA TSNE Isomap UMAP)
output_pattern='experiments/hyperparameter_search/real_world/{dataset}/{model}.json'
output_pattern_cycle='experiments/hyperparameter_search/real_world/{dataset}/{model}-cycle.json'


python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${ae_models[*]} \
  --name dataset --set MNIST FashionMNIST \
  --name dummy --set overrides.model__parameters__autoencoder_model=ConvolutionalAutoencoder_2D \
  --output-pattern ${output_pattern}

python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${ae_models[*]} \
  --name dataset --set MNIST FashionMNIST \
  --name dummy --set overrides.model__parameters__autoencoder_model=ConvolutionalAutoencoder_2D \
  --name dummy2 --set overrides.model__parameters__toposig_kwargs__use_cycles=True \
  --output-pattern ${output_pattern_cycle}


python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${competitor_methods[*]} \
  --name dataset --set MNIST FashionMNIST \
  --output-pattern ${output_pattern}
