# LVD_map.py

from joblib import Parallel, delayed
from eugene.src.tools.LVDSim import simData, tuplesToBlocks, energyDistanceMatrixParallel
import multiprocessing
from eugene.src.tools.alphaBetaGrid import *
from multiprocessing import cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def map_data(init_pops, trans_pops, caps, max_time, num_times, depth):
    num_cores = multiprocessing.cpu_count() - 2
    list_of_points = lv_map_points(alpha_steps=depth, beta_steps=depth)
    params = Parallel(n_jobs=num_cores)(delayed(point_to_param)(point, init_pops, trans_pops, caps) for point in list_of_points)

    data = Parallel(n_jobs=num_cores)(delayed(simData)([param], max_time, num_times, overlaid, stochastic_reps=None, range_cover=False) for param in tqdm(params))

    # data = simData(params, max_time, num_times, overlay, range_cover=False)
    new_data = []
    for c in data:
        new_data.append(c[0])

    blocks = tuplesToBlocks(new_data)

    dmat = energyDistanceMatrixParallel(blocks)
    return dmat


def point_to_param(point, init_pops, trans_pops, caps):
    sigma = [0.1, 0.1, 0.1, 0.1]
    param = [point[0], caps, point[1], sigma, init_pops, trans_pops]
    return param

def overlaid(x):
    return x

# def map_demo():
    #do something


if __name__ == '__main__':
    # overlay = lambda x: np.mean(x, axis=1)
    # overlay = lambda x: x
    k = np.array([100., 100., 100., 100.])
    init_pops = np.array([5., 5., 5., 5.])
    trans_pops = np.array([8., 8., 8., 8.])
    dist_mat = map_data(init_pops, trans_pops, k, 50., 50., 30.)
    plt.imshow(dist_mat, cmap='hot', interpolation='nearest')
    plt.show()
    np.savetxt("LV_map_dmat.txt", dist_mat, fmt='%.5f')
