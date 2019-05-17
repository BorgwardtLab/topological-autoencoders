
import unittest

import numpy as np

from src.topology import AlephPersistenHomologyCalculation, \
    PersistentHomologyCalculation


def generate_mock_data(n_instances, n_dim):
    x = np.random.normal(0., 1., size=(n_instances, n_dim))
    distances = np.linalg.norm(x[:, None] - x, axis=2)
    return x, distances



class Test0DSignature(unittest.TestCase):
    def test_0d_signature(self, n_instances=20, n_dim=50):
        # Sortig the tuples from aleph according to distances should be in
        # line with the python implementation."""
        x, distances = generate_mock_data(n_instances, n_dim)
        al_calc = AlephPersistenHomologyCalculation(
            compute_cycles=False, sort_selected=True)
        calc = PersistentHomologyCalculation()

        al_sig, _ = al_calc(distances)
        sig, _ = calc(distances)
        sucess = np.allclose(al_sig, sig)
        if not sucess:
            print('Aleph:')
            print(al_sig)
            print('Python:')
            print(sig)
        assert sucess
