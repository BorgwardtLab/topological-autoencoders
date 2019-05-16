# Topologically Constrained Autoencoder

## Installing `Aleph`

Provided that all dependencies are satisfied, this should be sufficient
to install the module:

    $ git submodule update --init
    $ cd Aleph
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make aleph
    $ cd ../../
    $ pipenv run install_aleph


## Calling makefile

The makefile automatically executes all experiments in the experiments folder
according to their highest level folder (e.g. experiments/train_model/xxx.json
calls exp.train_model with the config file experiments/train_model/xxx.json).

### Filtering
In order to only run a subset of experiments one can use the `filtered` target
in combination with the `FILTER` config option:
```bash
make filtered FILTER=train_model/synthetic_experiments
```

in order to execute all experiment in the `train_model/synthetic_experiments`
folder.


### Parallel execution
In order to run multiple experiments in parallel, one can pass the `-j` flag to
make:

```bash
make -j 2 filtered FILTER=train_model/synthetic_experiments
```


### Overrides
It is also possible to override sacred configs using the `SACRED_OVERRIDES`
variable:

```bash
make -j 2 filtered FILTER=train_model/synthetic_experiments SACRED_OVERRIDES='quiet=True'
```
### Creating config files

```bash
python scripts/configs_from_product.py exp.train_model \
  --name model \
  --set model.Vanilla model.TopoReg model.TopoRegVertex model.TopoRegEdge model.TopoRegEdgeSymmetric \
  --name dataset --set dataset.SwissRoll dataset.SCurve \
  --name dummy --set model.parameters.autoencoder_model=MLPAutoencoder \
  --name dummy2 --set n_epochs=100 \
  --output-pattern 'experiments/train_model/mlpautoencoder_dimred/{dataset}/{model}.json'
```

## Implementation TODOs:

* datasets
* topo sorting
* evaluation scenario
* hyperparameter search
