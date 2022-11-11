# file test_clustering.py

import unittest
from eugene.src.tools.clustering import *
import numpy as np


class TestClustering(unittest.TestCase):

    def setUp(self):
        self.distance_matrix = np.array([[0., 2., 1.5], [2., 0., 1.], [1.5, 2.,
            0.]])

    def test_matrix_to_orderings(self):
        rankings = matrix_to_orderings(self.distance_matrix) 
        assert rankings == [['r0', 'r2', 'r1'], ['r1', 'r2', 'r0'], ['r2', 'r0',
            'r1']]

    def test_combinations(self):
        items = ['a', 'b', 'c']
        out = combinations(items, 1)
        assert out == items
        out = combinations(items, 2)
        tmp = set(out)
        assert ('a', 'b') in tmp
        assert ('a', 'c') in tmp
        assert ('b', 'c') in tmp

if __name__ == '__main__': 
    unittest.main() 
