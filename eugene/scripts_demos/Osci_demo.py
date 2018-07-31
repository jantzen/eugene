# Osci_demo.py

from eugene.src.virtual_sys.damped_harmonic_oscillator import *
from tqdm import trange, tqdm
from eugene.src.tools.LVDSim import tuplesToBlocks, energyDistanceMatrixParallel
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np


def kind_dists(data):
    print(np.array(data).shape)
    stacked = []
    for sys in data:
        col0 = sys[0]
        col1 = sys[1]
        print(np.array(col0).shape)
        stacked.append(np.array([col0, col1]).T)
    print(np.array(stacked).shape)
    dmat = energyDistanceMatrixParallel(np.array(stacked))

    return dmat


def changing_zeta():
    overlay = lambda x: x

    zetas = []
    params = []
    for n in tqdm(np.arange(0.1, 2.0, 0.1)):
        zeta = n
        zetas.append(zeta)

        y0 = 1.0
        p0 = 0.0
        w0 = 2.0

        init1 = y0 - 0.05
        init_trans1 = y0

        params1 = [zeta, init1, p0, w0, init_trans1, p0, w0]
        params.append(params1)

    data = simData(params, 2.5, 1000.0, overlay, range_cover=False)
    print(params)
    print(data.shape)
    dmat = kind_dists(data)

    print(zetas)
    print(dmat)

    plt.imshow(dmat, interpolation='nearest')
    plt.show()

    return dmat


def changing_positions():
    overlay = lambda x: x

    pos = []
    params = []
    for n in tqdm(np.arange(-1.0, 1.0, 0.1)):
        y0 = n
        pos.append(y0)

        zeta = 0.2
        p0 = 0.0
        w0 = 2.0

        init1 = y0 - 0.05
        init_trans1 = y0

        params1 = [zeta, init1, p0, w0, init_trans1, p0, w0]
        params.append(params1)

    data = simData(params, 15., 10000, overlay, range_cover=False)
    dmat = kind_dists(data)

    print(pos)

    plt.imshow(dmat, interpolation='nearest')
    plt.show()

    return dmat


if __name__ == '__main__':
    changing_zeta()
    # changing_positions()