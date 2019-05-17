
import unittest

import numpy as np

from src.topology import AlephPersistenHomologyCalculation, \
    PersistentHomologyCalculation


def generate_mock_data(n_instances, n_dim):
    x = np.random.normal(0., 1., size=(n_instances, n_dim))
    distances = np.linalg.norm(x[:, None] - x, axis=2)
    return x, distances



class Test0DSignature(unittest.TestCase):
    def test_0D_signature_unsorted(self, n_instances=20, n_dim=50):
        x, distances = generate_mock_data(n_instances, n_dim)
        al_calc = AlephPersistenHomologyCalculation(compute_cycles=False)
        calc = PersistentHomologyCalculation()
        al_sig, _ = al_calc(distances)
        sig, _  = calc(distances)
        sucess = np.allclose(al_sig, sig)
        if not sucess:
            print('Aleph:')
            print(al_sig)
            print('Python:')
            print(sig)
        assert sucess

    def test_0D_signature_sorted(self, n_instances=20, n_dim=50):
        x, distances = generate_mock_data(n_instances, n_dim)
        al_calc = AlephPersistenHomologyCalculation(compute_cycles=False)
        calc = PersistentHomologyCalculation()
        al_sig, _ = al_calc(distances)
        sig, _  = calc(distances)
        al_sig = al_sig[np.lexsort((al_sig[:, 1], al_sig[:, 0]))]
        sig = sig[np.lexsort((sig[:, 1], sig[:, 0]))]
        sucess = np.allclose(al_sig, sig)
        if not sucess:
            print('Aleph:')
            print(al_sig)
            print('Python:')
            print(sig)
        assert sucess
