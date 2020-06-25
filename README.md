#Topological Autoencoder

In order to reproduce the results indicated in the paper simply setup an
environment using the provided `Pipfile` and `pipenv` and run the experiments
using the provided makefile:

```bash
pipenv install --skip-lock  
```

# Running a method:
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


## Using Aleph
In the paper, low-dimensional persistence homology calculations were performed in python directly. However, for higher dimensions, we recommend to use Aleph, a C++ library. We aim to better integrate this into this codebase, stay tuned!

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

