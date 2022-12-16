# file test_clustering.py

import unittest
from eugene.src.tools.clustering import *
import numpy as np


class TestClustering(unittest.TestCase):

    def setUp(self):
        pass


    def test_matrix_to_orderings(self):
        dm = np.array([[0., 2., 1.5], [2., 0., 1.], [1.5, 2.,0.]])
        rankings = matrix_to_orderings(dm) 
        assert rankings == [['r0', 'r2', 'r1'], ['r1', 'r2', 'r0'], ['r2', 'r0',
            'r1']]
        # test soft ranking
        dm = np.array([[0., 2., 2.1], [2., 0., 1.], [2.1, 2.,0.]])
        rankings = matrix_to_orderings(dm, epsilon=0.2)
        assert ['r0', 'r1', 'r1'] in rankings
        assert ['r0', 'r2', 'r2'] in rankings
        assert ['r1', 'r2', 'r0'] in rankings
        assert ['r2', 'r0', 'r0'] in rankings
        assert ['r2', 'r1', 'r1'] in rankings
        

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

    def test_remove_inconsistencies(self):
        clusters_found = [('a','b','c'), ('a','d','c'), ('c','b','a'),
                ('a','b','c','d')]
        clusters_found = remove_inconsistencies(clusters_found)
        assert clusters_found == [('a','b','c','d')]

        clusters_found = [('d','b'), ('a','e','c'), ('c','b','d'),
                ('a','b','c','d')]
        clusters_found = remove_inconsistencies(clusters_found)
        assert clusters_found == [('d','b')]

    def test_soft_qualitative_cluster(self):
        # soft clustering hierarchical example
        dm = np.array([[0., 1., 2.5, 6.], [1., 0., 0.1, 4.], [2.5, 0.1, 0., 5.],
            [6., 4., 5., 0.]])
        r = matrix_to_orderings(dm, ['a', 'b', 'c', 'd'], epsilon=0.5)
        out = qualitative_cluster(['a', 'b', 'c', 'd'], r)
        assert out == [('a', 'b', 'c')]

        # soft clustering large margin
        dm = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        r = matrix_to_orderings(dm, ['a', 'b', 'c'], epsilon=10.)
        out = qualitative_cluster(['a', 'b', 'c'], r)
        assert out == []
        

if __name__ == '__main__': 
    unittest.main() 
