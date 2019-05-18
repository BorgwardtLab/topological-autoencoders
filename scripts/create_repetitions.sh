#!/bin/bash

# Without cycles
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set model.Vanilla model.TopoRegSorted model.TopoRegEdgeSymmetric model.VAE model.VAETopoRegEdgeSymmetric model.VAETopoRegEdgeRandom \
  --name dataset --set dataset.SwissRoll dataset.SCurve \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder \
  --name dummy2 --set n_epochs=100 \
  --output-pattern 'experiments/train_model/dimred_repetitions/{dataset}/{model}_{rep}.json'

python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set model.Vanilla model.TopoRegSorted model.TopoRegEdgeSymmetric model.VAE model.VAETopoRegEdgeSymmetric model.VAETopoRegEdgeRandom \
  --name dataset --set dataset.Spheres \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder_Spheres \
  --name dummy2 --set n_epochs=100 \
  --output-pattern 'experiments/train_model/dimred_repetitions/{dataset}/{model}_{rep}.json'

# With cycles
python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set model.TopoRegSorted model.TopoRegEdgeSymmetric model.VAETopoRegEdgeSymmetric model.VAETopoRegEdgeRandom \
  --name dataset --set dataset.SwissRoll dataset.SCurve \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder \
  --name dummy2 --set n_epochs=100 \
  --name dummy3 --set model.parameters.toposig_kwargs.use_cycles=True \
  --output-pattern 'experiments/train_model/dimred_repetitions_cycles/{dataset}/{model}_{rep}.json'

python scripts/configs_from_product.py exp.train_model \
  --name rep --set rep1 rep2 rep3 rep4 \
  --name model \
  --set model.TopoRegSorted model.TopoRegEdgeSymmetric model.VAETopoRegEdgeSymmetric model.VAETopoRegEdgeRandom \
  --name dataset --set dataset.Spheres \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder_Spheres \
  --name dummy2 --set n_epochs=100 \
  --name dummy3 --set model.parameters.toposig_kwargs.use_cycles=True \
  --output-pattern 'experiments/train_model/dimred_repetitions_cycles/{dataset}/{model}_{rep}.json'
