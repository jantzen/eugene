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

        x_min = []
        x_max = []
        x_std = []
        y_min = []
        y_max = []
        y_std = []
        for block in blocks:
            x_min.append(np.min(block[:, 0]))
            x_max.append(np.max(block[:, 0]))
            x_std.append(np.std(block[:, 0]))
            y_min.append(np.min(block[:, 1]))
            y_max.append(np.max(block[:, 1]))
            y_std.append(np.std(block[:, 1]))
        x_std = np.max(x_std)
        x_min = np.min(x_min) - x_std
        x_max = np.max(x_max) + x_std
        y_std = np.max(y_std)
        y_min = np.min(y_min) - y_std
        y_max = np.max(y_max) + y_std

        densities = blocksToScipyDensities(blocks)

        dmat = distanceH2D(densities, x_range=[x_min, x_max],
                           y_range=[y_min, y_max])

        dist = dmat[0][1] - dmat[0][0]
        return dist


    def ave_pair_dist(self, data):

        dmat = AveHellinger(data)
        dist = dmat[0][1] - dmat[0][0]
        return dist

    def test_same_more_data(self):
        print ("test_same_more_data")

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([1, 1])
        init2 = init1

        init_trans1 = np.array([1.2, 1.2])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x+1)
            lens.append(n)
            data, low, high = simData([params1, params2], 5., n, overlay)
            dist = self.pair_dist(data)
            dists.append(dist)

        print(dists)

        self.assertTrue(dists[0] > dists[-1])


    def test_diff_more_data(self):
        print ("test_diff_more_data")

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([1, 1])
        init2 = init1

        init_trans1 = np.array([1.2, 1.2])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x+1)
            lens.append(n)
            data, low, high = simData([params1, params2], 5., n, overlay)
            dist = self.pair_dist(data)
            dists.append(dist)

        print (dists)

        self.assertTrue(dists[0] < dists[-1])


    @unittest.skip
    def test_same_more_info(self):
        print ("test_same_more_info")
        # currently out of commission as comparing more than one variable is...
        # difficult.

        r1 = np.array([1., 2.])
        r2 = 1.5 * r1

        k1 = k2 = np.array([100., 100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([0.5, 0.5])
        init2 = init1

        init_trans1 = np.array([0.8, 0.8])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        data, low, high = simData([params1, params2], 5., 500, overlay)

        blocks = tuplesToBlocks(data)

        rblocks = resampleToUniform(blocks, low, high)

        kdes = blocksToKDEs(rblocks)

        densities = KDEsToDensities(kdes)

        dmat = distanceH2D(densities, y_range=[low, high])

        dist1 = dmat[0][1] - dmat[0][0]

        less_overlay = lambda x: x

        more_data, more_low, more_high = simData([params1, params2], 5., 500, less_overlay)

        more_blocks = tuplesToBlocks(more_data)

        more_rblocks = resampleToUniform(more_blocks, more_low, more_high)

        more_kdes = blocksToKDEs(more_rblocks)

        more_densities = KDEsToDensities(more_kdes)

        dmat2 = distanceH2D(more_densities, y_range=[more_low, more_high])

        dist2 = dmat2[0][1] - dmat2[0][0]

        self.assertTrue(dist1 > dist2)


if __name__ == '__main__':
    unittest.main()
