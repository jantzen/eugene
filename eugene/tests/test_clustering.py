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
        items = ['a', 'b', 'c', 'd']
        out = combinations(items, 1)
        assert type(out) == list
        assert out == [('a',), ('b',), ('c',), ('d',)]
        out = combinations(items, 2)
        assert ('a', 'b') in out
        assert ('a', 'c') in out
        assert ('b', 'c') in out
        out = combinations(items, 2)
        assert ('a', 'b') in out
        assert ('a', 'c') in out
        assert ('a', 'd') in out
        assert ('b', 'c') in out
        assert ('b', 'd') in out
        out = combinations(items, 3)
        assert ('a', 'b', 'c') in out
        assert ('a', 'b', 'd') in out
        assert ('a', 'c', 'd') in out
        assert ('b', 'c', 'd') in out


    def test_check_if_cluster(self):
        rankings = [['r0', 'r2', 'r1'], ['r1', 'r2', 'r0'], ['r2', 'r0', 'r1']]
        assert check_if_cluster(('r0','r2'), rankings)
        rankings = [['r0', 'r1', 'r2'], ['r0', 'r2', 'r1'], ['r1', 'r0', 'r2']]
	assert not check_if_cluster(('r0','r2'), rankings)
        rankings = [['r0', 'r1', 'r2', 'r3'], ['r0', 'r3', 'r2', 'r1'], ['r1', 'r0', 'r2']]
	assert not check_if_cluster(('r0','r2'), rankings)


    def test_qualitative_cluster(self):
        points = ['r0', 'r1', 'r2']
        rankings = [['r0', 'r2', 'r1'], ['r1', 'r2', 'r0'], ['r2', 'r0', 'r1']]
        out = qualitative_cluster(points, rankings)
        assert out == [('r0', 'r2')]

        # two-cluster example
        dm1 = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        dm2 = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
        dm3 = np.array([[0., 2., 3.], [2., 0., 1.], [3., 1., 0.]])
        dm4 = np.array([[0., 3., 2.], [3., 0., 1.], [2., 1., 0.]])
	r1 = matrix_to_orderings(dm1, ['a', 'b', 'c'])
	r2 = matrix_to_orderings(dm2, ['a', 'b', 'd'])
	r3 = matrix_to_orderings(dm3, ['a', 'c', 'd'])
	r4 = matrix_to_orderings(dm4, ['b', 'c', 'd'])
	out = qualitative_cluster(['a', 'b', 'c', 'd'], r1 + r2 + r3 + r4)
	assert ('a', 'b') in out and ('c', 'd') in out

        # one-cluster example
        dm1 = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        dm2 = np.array([[0., 1., 3.], [2., 0., 1.], [3., 1., 0.]])
        dm3 = np.array([[0., 2., 3.], [2., 0., 1.], [3., 1., 0.]])
        dm4 = np.array([[0., 3., 2.], [3., 0., 1.], [2., 1., 0.]])
	r1 = matrix_to_orderings(dm1, ['a', 'b', 'c'])
	r2 = matrix_to_orderings(dm2, ['a', 'b', 'd'])
	r3 = matrix_to_orderings(dm3, ['a', 'c', 'd'])
	r4 = matrix_to_orderings(dm4, ['b', 'c', 'd'])
	out = qualitative_cluster(['a', 'b', 'c', 'd'], r1 + r2 + r3 + r4)
	assert not ('a', 'b') in out and ('c', 'd') in out

        # hierarchical example
	dm = np.array([[0., 1., 2., 4.], [1., 0., 2., 4.], [2., 2., 0., 3.],
	    [4., 4., 3., 0.]])
	r = matrix_to_orderings(dm, ['a', 'b', 'c', 'd'])
	out = qualitative_cluster(['a', 'b', 'c', 'd'], r)
	assert ('a', 'b') in out
	assert ('a', 'b', 'c') in out

if __name__ == '__main__': 
    unittest.main() 
