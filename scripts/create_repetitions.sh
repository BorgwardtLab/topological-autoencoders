#!/bin/bash

ae_models=(model.Vanilla model.TopoReg model.TopoRegSorted model.TopoRegEdgeSymmetric)
ae_models_cycles=(model.TopoReg model.TopoRegSorted model.TopoRegEdgeSymmetric)
vae_models=(model.VAE model.VAETopoReg model.VAETopoRegSorted model.VAETopoRegEdgeSymmetric)
vae_models_cycles=(model.VAETopoReg model.VAETopoRegSorted model.VAETopoRegEdgeSymmetric)
output_pattern='experiments/train_model/dimensionality_reduction/{dataset}/{model}_{rep}.json'
output_pattern_cycles='experiments/train_model/dimensionality_reduction/{dataset}/{model}-cycles_{rep}.json'


# Regular autoencoder without cycles
# Swissroll and SCruve
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${ae_models[*]} \
  --name dataset --set dataset.SwissRoll dataset.SCurve \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder \
  --name dummy2 --set n_epochs=100 \
  --output-pattern ${output_pattern}
# Spheres
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${ae_models[*]} \
  --name dataset --set dataset.Spheres \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder_Spheres \
  --name dummy2 --set n_epochs=100 \
  --output-pattern ${output_pattern}

# VAE without cycles
# Swissroll and SCruve
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${vae_models[*]} \
  --name dataset --set dataset.SwissRoll dataset.SCurve \
  --name dummy2 --set n_epochs=100 \
  --output-pattern ${output_pattern}
# Spheres
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${vae_models[*]} \
  --name dataset --set dataset.Spheres \
  --name dummy --set model.parameters.ae_kwargs.input_dim=101 \
  --name dummy2 --set n_epochs=100 \
  --output-pattern ${output_pattern}


# Regular autoencoder with cycles
# Swissroll and SCruve
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${ae_models_cycles[*]} \
  --name dataset --set dataset.SwissRoll dataset.SCurve \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder \
  --name dummy2 --set n_epochs=100 \
  --name dummy3 --set model.parameters.toposig_kwargs.use_cycles=True \
  --output-pattern ${output_pattern_cycles}
# Spheres
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${ae_models_cycles[*]} \
  --name dataset --set dataset.Spheres \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder_Spheres \
  --name dummy2 --set n_epochs=100 \
  --name dummy3 --set model.parameters.toposig_kwargs.use_cycles=True \
  --output-pattern ${output_pattern_cycles}

# VAE with cycles
# Swissroll and SCruve
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${vae_models_cycles[*]} \
  --name dataset --set dataset.SwissRoll dataset.SCurve \
  --name dummy2 --set n_epochs=100 \
  --name dummy3 --set model.parameters.toposig_kwargs.use_cycles=True \
  --output-pattern ${output_pattern_cycles}
# Spheres
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set ${vae_models_cycles[*]} \
  --name dataset --set dataset.Spheres \
  --name dummy --set model.parameters.ae_kwargs.input_dim=101 \
  --name dummy2 --set n_epochs=100 \
  --name dummy3 --set model.parameters.toposig_kwargs.use_cycles=True \
  --output-pattern ${output_pattern_cycles}
