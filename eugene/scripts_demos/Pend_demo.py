# Pend_demo.py

from eugene.src.virtual_sys.multi_pendulums import *
from tqdm import trange
from eugene.src.tools.LVDSim import AveHellinger
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    overlay = lambda x: np.mean(x, axis=1)

    dists = []
    lens = []
    for n in trange(10):
        l = 1.-(n / 10.)
        lens.append(l)

        pends = 2

        init1 = 135
        init_trans1 = 150
        init2 = 150
        init_trans2 = 165

        lengths = [1., l]

        params1 = [pends, init1, init_trans1, lengths]
        params2 = [pends, init2, init_trans2, lengths]

        data, low, high = simData([params1, params2], 10., 1000, overlay)
        dmat = AveHellinger(data)
        dist = dmat[0][1] - dmat[0][0]
        dists.append(dist)

    print(lens)
    print(dists)

    plt.plot(lens, dists)
    plt.show()