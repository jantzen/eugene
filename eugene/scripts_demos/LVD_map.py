# LVD_map.py

from eugene.src.tools.LVDSim import *
from eugene.src.tools.alphaBetaGrid import *
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np


def map_data(init_pops, trans_pops, caps, max_time, num_times, overlay, depth):
    num_cores = multiprocessing.cpu_count() - 2
    list_of_points = lv_map_points(alpha_steps=depth, beta_steps=depth)
    params = Parallel(n_jobs=num_cores)(delayed(point_to_param)(point, init_pops, trans_pops, caps) for point in list_of_points)

    data = simData(params, max_time, num_times, overlay, range_cover=False)

    blocks = tuplesToBlocks(data)

    dmat = energyDistanceMatrixParallel(blocks)
    return dmat


def point_to_param(point, init_pops, trans_pops, caps):
    param = [point[0], caps, point[1], init_pops, trans_pops]
    return param


# def map_demo():
    #do something


if __name__ == '__main__':
    # overlay = lambda x: np.mean(x, axis=1)
    overlay = lambda x: x
    k = np.array([100., 100., 100., 100.])
    init_pops = np.array([5., 5., 5., 5.])
    trans_pops = np.array([8., 8., 8., 8.])
    dmat = map_data(init_pops, trans_pops, k, 100., 1000., overlay, 500.)
    plt.imshow(dmat, cmap='Blues', interpolation='nearest')
    plt.show()
    np.savetxt("LVD_map_dmat.txt", dmat, fmt='%.5f')