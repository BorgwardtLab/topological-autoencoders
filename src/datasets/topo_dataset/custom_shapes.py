import numpy as np
from tadasets.dimension import embed

'''
source: https://github.com/scikit-tda/tadasets/blob/master/tadasets/shapes.py
We modify the module here locally, s.t. all shapes return a conitnuous label for nicer visualization
'''

__all__ = ["torus", "dsphere", "sphere", "swiss_roll", "infty_sign"]


## TODO: Make a base class that controls `ambient` and `noise`.
class Shape:
    def __init__(self):
        pass


def dsphere(n=100, d=2, r=1, noise=None, ambient=None):
    """
    Sample `n` data points on a d-sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """
    data = np.random.randn(n, d+1)

    # Normalize points to the sphere
    data = r * data / np.sqrt(np.sum(data**2, 1)[:, None]) 

    if noise: 
        data += noise * np.random.randn(*data.shape)

    if ambient:
        assert ambient > d, "Must embed in higher dimensions"
        data = embed(data, ambient)



    return data


def sphere(n=100, r=1, noise=None, ambient=None):
    """
        Sample `n` data points on a sphere.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    r : float
        Radius of sphere.
    ambient : int, default=None
        Embed the sphere into a space with ambient dimension equal to `ambient`. The sphere is randomly rotated in this high dimensional space.
    """

    theta = np.random.random((n,)) * 2.0 * np.pi
    phi = np.random.random((n,)) * np.pi
    rad = np.ones((n,)) * r

    data = np.zeros((n, 3))

    data[:, 0] = rad * np.cos(theta) * np.cos(phi)
    data[:, 1] = rad * np.cos(theta) * np.sin(phi)
    data[:, 2] = rad * np.sin(theta)


    if noise: 
        data += noise * np.random.randn(*data.shape)

    if ambient:
        data = embed(data, ambient)

    return data, theta


def torus(n=100, c=2, a=1, noise=None, ambient=None):
    """
    Sample `n` data points on a torus.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    c : float
        Distance from center to center of tube.
    a : float
        Radius of tube.
    ambient : int, default=None
        Embed the torus into a space with ambient dimension equal to `ambient`. The torus is randomly rotated in this high dimensional space.
    """

    assert a <= c, "That's not a torus"

    theta = np.random.random((n,)) * 2.0 * np.pi
    phi = np.random.random((n,)) * 2.0 * np.pi

    data = np.zeros((n, 3))
    data[:, 0] = (c + a * np.cos(theta)) * np.cos(phi)
    data[:, 1] = (c + a * np.cos(theta)) * np.sin(phi)
    data[:, 2] = a * np.sin(theta)

    if noise: 
        data += noise * np.random.randn(*data.shape)

    if ambient:
        data = embed(data, ambient)

    return data, theta


def swiss_roll(n=100, r=10, noise=None, ambient=None):
    """Swiss roll implementation

    Parameters
    ----------
    n : int 
        Number of data points in shape.
    r : float
        Length of roll
    ambient : int, default=None
        Embed the swiss roll into a space with ambient dimension equal to `ambient`. The swiss roll is randomly rotated in this high dimensional space.

    References
    ----------
    Equations mimic [Swiss Roll and SNE by jlmelville](https://jlmelville.github.io/smallvis/swisssne.html)
    """

    phi = (np.random.random((n,)) * 3 + 1.5) * np.pi
    psi = np.random.random((n,)) * r

    data = np.zeros((n, 3))
    data[:, 0] = phi * np.cos(phi)
    data[:, 1] = phi * np.sin(phi)
    data[:, 2] = psi

    if noise: 
        data += noise * np.random.randn(*data.shape)

    if ambient:
        data = embed(data, ambient)

    return data, phi


def infty_sign(n=100, noise=None):
    """Construct a figure 8 or infinity sign with :code:`n` points and noise level with :code:`noise` standard deviation.

    Parameters
    ============

    n: int
        number of points in returned data set.
    noise: float
        standard deviation of normally distributed noise added to data.
    
    """


    t = np.linspace(0, 2*np.pi, n+1)[0:n]
    X = np.zeros((n, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(2*t)

    if noise:
        X += noise * np.random.randn(n, 2)
    
    return X
