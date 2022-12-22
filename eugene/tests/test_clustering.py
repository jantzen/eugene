# file test_clustering.py

import unittest
from eugene.src.tools.clustering import *
import numpy as np


class TestClustering(unittest.TestCase):

    def setUp(self):
        pass


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


    def test_find_greatest_separation(self):
        matrix = np.array([[0., 1., 2., 4.], [1., 0., 2., 4.], [2., 2., 0., 3.],
            [4., 4., 3., 0.]])
        dm = DistanceMatrix(matrix, ['a','b','c','d'])
        gs = find_greatest_separation(dm, ('a','b','c'))
        assert gs == 2.0
        gs = find_greatest_separation(dm, ('a','b','d'))
        assert gs == 4.0


    def test_check_if_cluster(self):
        matrix = np.array([[0., 1., 2., 4.], [1., 0., 2., 4.], [2., 2., 0., 3.],
            [4., 4., 3., 0.]])
        dm = DistanceMatrix(matrix, ['a','b','c','d'])
        assert check_if_cluster(('a', 'b'), [dm], ['a','b','c','d'])
        assert not check_if_cluster(('a', 'd'), [dm], ['a','b','c','d'])
        assert not check_if_cluster(('a', 'b'), [dm], ['a','b','c','d'],
                epsilon=3.)
        matrix1 = np.array([[0., 1., 2.], [1., 0., 2.], [2., 2., 0.]])
        matrix2 = np.array([[0., 2., 4.], [2., 0., 3.], [4., 3., 0.]])
        matrix3 = np.array([[0., 2., 4.], [2., 0., 3.], [4., 3., 0.]])
        dm1 = DistanceMatrix(matrix1, ['a','b','c'])
        dm2 = DistanceMatrix(matrix2, ['b','c','d'])
        dm3 = DistanceMatrix(matrix3, ['a','c','d'])
        assert check_if_cluster(('a', 'b'), [dm1, dm2, dm3], ['a','b','c','d'])
        assert not check_if_cluster(('a', 'd'), [dm1, dm2, dm3], ['a','b','c','d'])
        assert not check_if_cluster(('a', 'b'), [dm1, dm2, dm3],
                ['a','b','c','d'], epsilon=3)


#    def test_check_if_cluster(self):
#        rankings = [['r0', 'r2', 'r1'], ['r1', 'r2', 'r0'], ['r2', 'r0', 'r1']]
#        assert check_if_cluster(('r0','r2'), rankings)
#        rankings = [['r0', 'r1', 'r2'], ['r0', 'r2', 'r1'], ['r1', 'r0', 'r2']]
#        assert not check_if_cluster(('r0','r2'), rankings)
#        rankings = [['r0', 'r1', 'r2', 'r3'], ['r0', 'r3', 'r2', 'r1'], ['r1', 'r0', 'r2']]
#        assert not check_if_cluster(('r0','r2'), rankings)
#

    def test_qualitative_cluster(self):
#        points = ['r0', 'r1', 'r2']
#        rankings = [['r0', 'r2', 'r1'], ['r1', 'r2', 'r0'], ['r2', 'r0', 'r1']]
#        out = qualitative_cluster(points, rankings)
#        assert out == [('r0', 'r2')]
#
        # two-cluster example
        matrix1 = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        matrix2 = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
        matrix3 = np.array([[0., 2., 3.], [2., 0., 1.], [3., 1., 0.]])
        matrix4 = np.array([[0., 3., 2.], [3., 0., 1.], [2., 1., 0.]])
        dm1 = DistanceMatrix(matrix1, ['a', 'b', 'c'])
        dm2 = DistanceMatrix(matrix2, ['a', 'b', 'd'])
        dm3 = DistanceMatrix(matrix3, ['a', 'c', 'd'])
        dm4 = DistanceMatrix(matrix4, ['b', 'c', 'd'])
        out = qualitative_cluster([dm1, dm2, dm3, dm4], ['a', 'b', 'c', 'd'])
        assert ('a', 'b') in out and ('c', 'd') in out

        # two-cluster example with different scales for each dm
        matrix1 = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        matrix2 = np.array([[0., 0.1, 0.3], [0.1, 0., 0.2], [0.3, 0.2, 0.]])
        matrix3 = np.array([[0., 4., 6.], [4., 0., 2.], [6., 2., 0.]])
        matrix4 = np.array([[0., 3., 2.], [3., 0., 1.], [2., 1., 0.]])
        dm1 = DistanceMatrix(matrix1, ['a', 'b', 'c'])
        dm2 = DistanceMatrix(matrix2, ['a', 'b', 'd'])
        dm3 = DistanceMatrix(matrix3, ['a', 'c', 'd'])
        dm4 = DistanceMatrix(matrix4, ['b', 'c', 'd'])
        out = qualitative_cluster([dm1, dm2, dm3, dm4], ['a', 'b', 'c', 'd'])
        assert ('a', 'b') in out and ('c', 'd') in out

        # one-cluster example
        matrix1 = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        matrix2 = np.array([[0., 1., 3.], [2., 0., 1.], [3., 1., 0.]])
        matrix3 = np.array([[0., 2., 3.], [2., 0., 1.], [3., 1., 0.]])
        matrix4 = np.array([[0., 3., 2.], [3., 0., 1.], [2., 1., 0.]])
        dm1 = DistanceMatrix(matrix1, ['a', 'b', 'c'])
        dm2 = DistanceMatrix(matrix2, ['a', 'b', 'd'])
        dm3 = DistanceMatrix(matrix3, ['a', 'c', 'd'])
        dm4 = DistanceMatrix(matrix4, ['b', 'c', 'd'])
        out = qualitative_cluster([dm1, dm2, dm3, dm4], ['a', 'b', 'c', 'd'])
        assert not ('a', 'b') in out and ('c', 'd') in out

        # hierarchical example
        matrix = np.array([[0., 1., 2., 4.], [1., 0., 2., 4.], [2., 2., 0., 3.],
            [4., 4., 3., 0.]])
        dm = DistanceMatrix(matrix,['a', 'b', 'c', 'd']) 
        out = qualitative_cluster([dm], ['a', 'b', 'c', 'd'])
        assert ('a', 'b') in out
        assert ('a', 'b', 'c') in out

        # soft clustering hierarchical example
        matrix = np.array([[0., 1., 2., 6.], [1., 0., 0.5, 4.], [2., 0.5, 0., 5.],
            [6., 4., 5., 0.]])
        dm = DistanceMatrix(matrix,['a', 'b', 'c', 'd']) 
        out = qualitative_cluster([dm], ['a', 'b', 'c', 'd'], epsilon=1.5)
        assert out == [('a', 'b', 'c')]

        # soft clustering large margin
        matrix = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        dm = DistanceMatrix(matrix,['a', 'b', 'c']) 
        out = qualitative_cluster([dm], ['a', 'b', 'c'], epsilon=10.)
        assert out == []
        
        # soft clustering large margin with relative epsilon
        matrix = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
        dm = DistanceMatrix(matrix,['a', 'b', 'c']) 
        out = qualitative_cluster([dm], ['a', 'b', 'c'], epsilon=1.,
                relative_epsilon=True)
        assert out == []

if __name__ == '__main__': 
    unittest.main() 
