# Pend_demo.py

from eugene.src.virtual_sys.multi_pendulums import *
from eugene.src.tools.LVDSim import tuplesToBlocks
from tqdm import trange
from eugene.src.tools.LVDSim import energyDistanceMatrixParallel
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np

if __name__ != '__main__':
    # overlay = lambda x: np.mean(x, axis=1)
    overlay = lambda x: x

    dists = []
    lens = []
    params = []
    for n in trange(100):
        l = 1.-(n / 100.)
        lens.append(l)

        pends = 2

        init1 = 135
        init_trans1 = 150
        # init2 = 150
        # init_trans2 = 165

        lengths = [1., l]

        params.append([pends, init1, init_trans1, lengths])
        # params2 = [pends, init2, init_trans2, lengths]

    data = simData(params, 10., 1000, overlay)
    dmat = energyDistanceMatrixParallel(data)
    # dist = dmat[0][1] - dmat[0][0]
    # dists.append(dist)

    # print(lens)
    # print(dists)

    # plt.plot(lens, dists)
    plt.imshow(dmat, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    overlay = lambda x: np.mean(x, axis=1)
    # overlay = lambda x: x[0]

    dists = []
    angle = []
    params = []
    for n in trange(180):
        l = n
        angle.append(l)

        pends = 2

        init1 = l
        init_trans1 = l+2

        lengths = [1., 1.]

        params.append([pends, init1, init_trans1, lengths])
        # params2 = [pends, init2, init_trans2, lengths]

    data = simData(params, 60., 1000, overlay)
    blocks = tuplesToBlocks(data)
    dmat = energyDistanceMatrixParallel(data)
    # dist = dmat[0][1] - dmat[0][0]
    # dists.append(dist)

    # print(lens)
    # print(dists)

    # plt.plot(lens, dists)
    plt.imshow(dmat, cmap='hot', interpolation='nearest')
    plt.show()