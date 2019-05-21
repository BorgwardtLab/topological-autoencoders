
import unittest

import numpy as np

from src.evaluation.measures_optimized import MeasureCalculator
import src.evaluation.measures as measures_comp


def generate_random_data():
    X = np.random.normal(size=(40, 20))
    Z = np.random.normal(size=(40, 2))
    return X, Z


class TestMeasures(unittest.TestCase):
    def test_k_independent(self):
        X, Z = generate_random_data()
        calc = MeasureCalculator(X, Z, 5)
        measures = calc.compute_k_independent_measures()

        for key, value in measures.items():
            fn = getattr(measures_comp, key)
            value_comp = fn(X, Z)
            assert value == value_comp

    def test_k_dependent(self):
        X, Z = generate_random_data()
        k_max = 25
        calc = MeasureCalculator(X, Z, k_max)
        for k in range(1, k_max+1):
            measures = calc.compute_k_dependent_measures(k)
            for key, value in measures.items():
                fn = getattr(measures_comp, key)
                value_comp = fn(X, Z, k)
                assert value == value_comp
