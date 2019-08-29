"""Competitor dimensionality reduction algorithms."""
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from umap import UMAP

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE

except ImportError:
    from sklearn.manifold import TSNE
