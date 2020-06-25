#!/bin/bash

ae_models=(Vanilla TopoRegEdgeSymmetric)
competitor_methods=(PCA TSNE Isomap UMAP)
output_pattern='experiments/hyperparameter_search/dimensionality_reduction/{dataset}/{model}.json'

# Autoencoder based
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${ae_models[*]} \
  --name dataset  \
  --set Spheres \
  --name dummy --set overrides.model__parameters__autoencoder_model=MLPAutoencoder_Spheres \
  --output-pattern ${output_pattern}

# Competitor
python scripts/configs_from_product.py exp.hyperparameter_search \
  --name model \
  --set ${competitor_methods[*]} \
  --name dataset --set Spheres \
  --output-pattern ${output_pattern}
