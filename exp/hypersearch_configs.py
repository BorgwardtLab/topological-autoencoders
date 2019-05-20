def MNIST():
    overrides = {'dataset__name': 'MNIST'}

def FashionMNIST():
    overrides = {'dataset__name': 'FashionMNIST'}

def SCurve():
    overrides = {'dataset__name': 'SCurve'}

def SwissRoll():
    overrides = {'dataset__name': 'SwissRoll'}

def Spheres():
    overrides = {'dataset__name': 'Spheres'}

def add_datasets(experiment):
    experiment.named_config(MNIST)
    experiment.named_config(FashionMNIST)
    experiment.named_config(SCurve)
    experiment.named_config(SwissRoll)
    experiment.named_config(Spheres)

def Vanilla():
    overrides = {'model__name': 'VanillaAutoencoderModel'}

def TopoReg():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.01, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
    }

def TopoRegVertex():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.01, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
        'model__parameters__toposig_kwargs__sort_selected': True,
    }

def TopoRegEdgeSymmetric():
    hyperparameter_space = {
        'model__parameters__lam': ('Real', 0.01, 10, 'log-uniform')
    }
    overrides = {
        'model__name': 'TopologicallyRegularizedAutoencoder',
        'model__parameters__toposig_kwargs__match_edges': 'symmetric',
    }

def add_models(experiment):
    experiment.named_config(Vanilla)
    experiment.named_config(TopoReg)
    experiment.named_config(TopoRegVertex)
    experiment.named_config(TopoRegEdgeSymmetric)

def PCA():
    overrides = {
        'model__name': 'PCA',
    }

def TSNE():
    overrides = {
        'model__name': 'TSNE'
    }

def UMAP():
    overrides = {
        'model__name': 'UMAP'
    }

def add_competitors(experiment):
    experiment.named_config(PCA)
    experiment.named_config(TSNE)
    experiment.named_config(UMAP)

