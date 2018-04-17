# Pend_demo.py

from eugene.src.virtual_sys.multi_pendulums import *
from tqdm import trange
from eugene.src.tools.LVDSim import tuplesToBlocks, energyDistanceMatrixParallel
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np


def kind_dists(data):
    blocks = tuplesToBlocks(data)

    dmat = energyDistanceMatrixParallel(blocks)

    return dmat


def changing_angles():
    overlay = lambda x: x

    thetas = []
    params = []
    for n in trange(180):
        theta = n
        thetas.append(theta)

        pends = 2

        init1 = theta + 5
        init_trans1 = theta

        lengths = [1., 1.]

        params1 = [pends, init1, init_trans1, lengths]
        params.append(params1)

    data = simData(params, 5., 1000, overlay, range_cover=False)

    dmat = kind_dists(data)

    print(thetas)
    print(dmat)

    plt.imshow(dmat, interpolation='nearest')
    plt.show()

    return dmat

def changing_lengths():
    overlay = lambda x: x

    dists = []
    lens = []
    params = []
    for n in trange(100):
        l = 1. - (n / 100.)
        lens.append(l)

        pends = 2

        init1 = 135
        init_trans1 = 140
        init2 = 160
        init_trans2 = 165

        lengths = [1., l]

        params1 = [pends, init1, init_trans1, lengths]
        params.append(params1)

    data = simData(params, 20., 10000, overlay, range_cover=False)
    dmat = kind_dists(data)

    print(lens)

    plt.imshow(dmat, interpolation='nearest')
    plt.show()

    return dists


if __name__ == '__main__':
    # changing_angles()
    changing_lengths()