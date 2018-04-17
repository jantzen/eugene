# test_robustness.py

import numpy as np
import matplotlib.pyplot as plt
from eugene.src.tools.LVDSim import *
from eugene.src.tools.StochasticTools import *
import unittest

# TODO: create toolbox method to take 2 data sets and produce distances.


class TestRobustness(unittest.TestCase):

    def pair_dist(self, data):
        blocks = tuplesToBlocks(data)

        dmat = energyDistanceMatrixParallel(blocks)

        return dmat[0][1]


    def ave_pair_dist(self, data):

        dmat = energyDistanceMatrixParallel(data)
        dist = dmat[0][1] - dmat[0][0]
        return dist


    @unittest.skip
    def test_same_more_data(self):
        print ("test_same_more_data")

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5, 5])
        init2 = init1

        init_trans1 = np.array([8, 8])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data_noisy(self):
        print("test_same_more_data_noisy")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5, 5])
        init2 = init1

        init_trans1 = np.array([8, 8])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        stdev = 1.0
        for x in trange(9):
            n = 10.0 * np.power(2, x + 1)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay, range_cover=False)
            noisy_data = []
            for tup in data:
                ntup = []
                for curve in tup:
                    ncurve = []
                    for point in curve:
                        ncurve.append(point + np.random.normal(0., stdev))
                    ntup.append(ncurve)
                noisy_data.append(ntup)

            data = noisy_data
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data_stochastic(self):
        print ("test_same_more_data_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x+1)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay, stochastic_reps=10, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_reps_stochastic(self):
        print ("test_same_more_reps_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        for x in trange(10):
            data = simData([params1, params2], 5., 500, overlay, stochastic_reps=x+1, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_noise_stochastic(self):
        print ("test_same_more_noise_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        for x in trange(10):
            sigma = [0.1, 0.1]*(x + 1)
            params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
            params2 = [r2, k2, alpha2, sigma, init2, init_trans2]
            data = simData([params1, params2], 5., 500, overlay, stochastic_reps=10, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data(self):
        print ("test_diff_more_data")

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x+1)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print (dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data_noisy(self):
        print("test_diff_more_data_noisy")

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        stdev = 1.0
        for x in trange(9):
            n = 10.0 * np.power(2, x + 1)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay, range_cover=False)
            noisy_data = []
            for tup in data:
                ntup = []
                for curve in tup:
                    ncurve = []
                    for point in curve:
                        ncurve.append(point + np.random.normal(0., stdev))
                    ntup.append(ncurve)
                noisy_data.append(ntup)

            data = noisy_data
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data_stochastic(self):
        print ("test_diff_more_data_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(10):
            n = 10.0 * np.power(2, x+1)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay, stochastic_reps=10, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print (dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_reps_stochastic(self):
        print ("test_diff_more_reps_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        for x in trange(10):
            data = simData([params1, params2], 5., 500, overlay, stochastic_reps=x+1, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print (dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_noise_stochastic(self):
        print ("test_diff_more_noise_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        for x in trange(10):
            sigma = [0.1, 0.1]*(x + 1)
            params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
            params2 = [r2, k2, alpha2, sigma, init2, init_trans2]
            data = simData([params1, params2], 5., 500, overlay, stochastic_reps=10, range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print (dists)
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() < dists2.var())


    @unittest.skip
    def test_same_more_info(self):
        print ("test_same_more_info")
        # currently in commission as comparing more than one variable is...
        # easy!

        r1 = np.array([1., 2.])
        r2 = 1.5 * r1

        k1 = k2 = np.array([100., 100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay,
                           range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data = simData([params1, params2], 5., n, less_overlay,
                           range_cover=False)
            dist = self.pair_dist(data)
            dists2.append(dist)

        print(dists2)

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    @unittest.skip
    def test_same_more_info(self):
        print ("test_same_more_info")

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data = simData([params1, params2], 5., n, overlay,
                           range_cover=False)
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data = simData([params1, params2], 5., n, less_overlay,
                           range_cover=False)
            dist = self.pair_dist(data)
            dists2.append(dist)

        print(dists2)

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    def test_diff_more_data_by_more_reps_stochastic(self):
        print("test_diff_more_data_by_more_reps_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        variances = np.zeros((10, 10))
        for x in trange(10):
            n = 10.0 * np.power(2, x+1)
            inner_dists = []
            for y in trange(10):
                data = simData([params1, params2], 5., n, overlay,
                               stochastic_reps=y + 1, range_cover=False)
                dist = self.pair_dist(data)
                inner_dists.append(dist)
                variances[x, y] = np.var(dist)
            dists.append(inner_dists)

        print(dists)
        plt.imshow(variances, interpolation='nearest')
        self.assertTrue(variances[0,0] > variances[8, 8])


if __name__ == '__main__':
    unittest.main()
