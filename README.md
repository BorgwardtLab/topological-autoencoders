# Topological Autoencoders

<img src="animations/topoae.gif" width="400"> <img src="animations/vanilla.gif" width="400">

## Reference

Please use the following BibTeX code to cite our [paper](https://arxiv.org/abs/1906.00722),
which is accepted for presentation at [ICML 2020](https://icml.cc/Conferences/2020):

```
@InProceedings{Moor20Topological,
  author        = {Moor, Michael and Horn, Max and Rieck, Bastian and Borgwardt, Karsten},
  title         = {Topological Autoencoders},
  year          = {2020},
  eprint        = {1906.00722},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  booktitle     = {Proceedings of the 37th International Conference on Machine Learning~(ICML)},
  series        = {Proceedings of Machine Learning Research},
  publisher     = {PMLR},
  volume        = {119},
  editor        = {Hal Daumé III and Aarti Singh},
  pages         = {7045--7054},
  abstract      = {We propose a novel approach for preserving topological structures of the input space in latent representations of autoencoders. Using persistent homology, a technique from topological data analysis, we calculate topological signatures of both the input and latent space to derive a topological loss term. Under weak theoretical assumptions, we construct this loss in a differentiable manner, such that the encoding learns to retain multi-scale connectivity information. We show that our approach is theoretically well-founded and that it exhibits favourable latent representations on a synthetic manifold as well as on real-world image data sets, while preserving low reconstruction errors.},
  pdf           = {http://proceedings.mlr.press/v119/moor20a/moor20a.pdf},
  url           = {http://proceedings.mlr.press/v119/moor20a.html},
}
```  

## Setup
In order to reproduce the results indicated in the paper simply setup an
environment using the provided `Pipfile` and `pipenv` and run the experiments
using the provided makefile:

```bash
pipenv install --skip-lock  
```

Alternatively, the exact versions used in this project can be accessed in ```requirements.txt```, however
this pip freeze contains a superset of all necessary libraries. To install it, run
```bash
pipenv install -r requirements.txt --skip-lock
```
  
## Running a method:
```bash
python -m exp.train_model with experiments/train_model/best_runs/Spheres/TopoRegEdgeSymmetric.json device='cuda' 
```
We used device='cuda', alternatively, if no gpu is available, use device='cpu'.

The above command trains our proposed method on the Spheres Data set. For different methods or datasets
simply adjust the last two directories of the path according to the directory structure.


## Calling makefile
The makefile automatically executes all experiments in the experiments folder
according to their highest level folder (e.g. experiments/train_model/xxx.json
calls exp.train_model with the config file experiments/train_model/xxx.json)
and writes the outputs to exp_runs/train_model/xxx/

For this use:
```bash
make filtered FILTER=train_model/repetitions
```
to run the test evaluations (repetitions) of the deep models
and for remaining baselines:
```bash
make filtered FILTER=fit_competitor/repetitions
```

We created testing repetitions by using the config from the best runs of the hyperparameter search (stored in best_runs/)


The models found in `train_model` correspond to neural network architectures.  

## Using Aleph (optional)

In the paper, low-dimensional persistent homology calculations are
implemented in Python directly. However, for higher dimensions, we
recommend to use Aleph, a C++ library. We aim to better integrate this
library into our code base, stay tuned!

Provided that all dependencies are satisfied, the following instructions should be sufficient
to install the module:

    $ git submodule update --init
    $ cd Aleph
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make aleph
    $ cd ../../
    $ pipenv run install_aleph

