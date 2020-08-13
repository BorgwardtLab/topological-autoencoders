#!/bin/bash

# Autoencoder based
output_pattern='experiments/train_model/repetitions/{rep}/{config}'

python scripts/configs_from_product.py exp.train_model \
  --name config \
  --set \
    experiments/train_model/best_runs/FashionMNIST/LinearAE-TopoRegEdgeSymmetric.json \
    experiments/train_model/best_runs/MNIST/LinearAE-TopoRegEdgeSymmetric.json \
    experiments/train_model/best_runs/Spheres/LinearAE-TopoRegEdgeSymmetric.json \
    experiments/train_model/best_runs/CIFAR/LinearAE-TopoRegEdgeSymmetric.json \
  --name rep --set rep1 rep2 rep3 rep4 rep5 \
  --name dummy --set evaluation.active=True \
  --name dummy2 --set evaluation.evaluate_on='test' \
  --output-pattern ${output_pattern}

for r in rep1 rep2 rep3 rep4 rep5;
do
    for d in FashionMNIST MNIST Spheres CIFAR;
    do
   mv experiments/train_model/repetitions/$r/experiments/train_model/best_runs/$d/* experiments/train_model/repetitions/$r/$d && rm -r experiments/train_model/repetitions/$r/experiments/train_model/best_runs/$d
    done

done

