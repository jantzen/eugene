# LVD_map.py

from joblib import Parallel, delayed
from eugene.src.tools.LVDSim import simDataAlt, tuplesToBlocks, energyDistanceMatrixParallel
import multiprocessing
from eugene.src.tools.alphaBetaGrid import *
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import sys, os, pickle


def map_data(init_pops, trans_pops, caps, max_time, num_times, depth):
    num_cores = multiprocessing.cpu_count() - 2
    list_of_points = lv_map_points(alpha_steps=depth, beta_steps=depth)

    # create a session folder

    params = Parallel(n_jobs=num_cores,verbose=5)(delayed(point_to_param)(point, init_pops, trans_pops, caps) for point in list_of_points)

    data = Parallel(n_jobs=num_cores,verbose=5)(delayed(simDataAlt)([param], max_time, num_times, overlaid, stochastic_reps=5) for param in params)

    # data = simData(params, max_time, num_times, overlay, range_cover=False)
    new_data = []
    for c in data:
        new_data.append(c[0])

    blocks = tuplesToBlocks(new_data)

    # clean data
    bad_data = []
    for i, block in enumerate(blocks):
        if not np.all(np.isfinite(block)):
            print("Bad data detected for params {}.".format(params[i]))
            bad_data.append([params[i], block])
            blocks[i] = blocks[i][~np.isnan(blocks[i]).any(axis=1)]

    dmat = energyDistanceMatrixParallel(blocks)
    return dmat, bad_data


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

    if len(sys.argv) < 2:
        folder = raw_input("Please enter a folder in which to save results: \n")
    else:
        folder = sys.argv[1]
    assert os.path.isdir(folder), "Folder does not exist."

    if not folder[-1] == '/':
        folder = folder + '/'

    k = np.array([100., 100., 100., 100.])
    init_pops = np.array([5., 5., 5., 5.])
    trans_pops = np.array([8., 8., 8., 8.])
    dist_mat, bad_data = map_data(init_pops, trans_pops, k, 10., 100., 100.)
    plt.imshow(dist_mat, cmap='hot', interpolation='nearest')
    np.savetxt(folder+"LVD_map_dmat.txt", dist_mat, fmt='%.5f')
    bad_file = open(folder+"LVD_map_bad_data.pkl",'wb')
    pickle.dump(bad_data, bad_file)
    plt.show()
