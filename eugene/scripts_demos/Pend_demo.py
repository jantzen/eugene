# Pend_demo.py

from eugene.src.virtual_sys.multi_pendulums import *
from tqdm import trange
from eugene.src.tools.LVDSim import tuplesToBlocks, blocksToScipyDensities, \
    distanceH2D
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np


def kind_dists(data):
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

    return dmat


def changing_angles():
    overlay = lambda x: x[0]

    thetas = []
    params = []
    for n in trange(10):
        theta = 160 - (n * 7)
        thetas.append(theta)

        pends = 2

        init1 = theta + 5
        init_trans1 = theta

        lengths = [1., 1.]

        params1 = [pends, init1, init_trans1, lengths]
        params.append(params1)

    data, low, high = simData(params, 5., 1000, overlay)
    dmat = kind_dists(data)

    print(thetas)
    print(dmat)
    return dmat

def changing_lengths():
    overlay = lambda x: x[0]

    dists = []
    lens = []
    for n in trange(10):
        l = 1. - (n / 10.)
        lens.append(l)

        pends = 2

        init1 = 135
        init_trans1 = 140
        init2 = 160
        init_trans2 = 165

        lengths = [1., l]

        params1 = [pends, init1, init_trans1, lengths]
        params2 = [pends, init2, init_trans2, lengths]

        data, low, high = simData([params1, params2], 20., 10000, overlay)
        dmat = kind_dists(data)
        dist = dmat[0][1] - dmat[0][0]
        dists.append(dist)

    print(lens)
    print(dists)

    plt.plot(lens, dists)
    plt.show()


if __name__ == '__main__':
    changing_angles()
    # changing_lengths()