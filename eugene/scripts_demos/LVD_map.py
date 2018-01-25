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

    data, low, high = simData(params, max_time, num_times, overlay)

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
    x_min = np.min(x_min)
    x_max = np.max(x_max)
    y_std = np.max(y_std)
    y_min = np.min(y_min)
    y_max = np.max(y_max)

    densities = blocksToScipyDensities(blocks)

    dmat = distanceH2D(densities, x_range=[x_min, x_max],
                       y_range=[y_min, y_max])
    return dmat


def point_to_param(point, init_pops, trans_pops, caps):
    param = [point[0], caps, point[1], init_pops, trans_pops]
    return param


def map_dmat(data, low, high):
    dmat = AveHellinger(data)
    return dmat


# def map_demo():
    #do something


if __name__ == '__main__':
    overlay = lambda x: np.mean(x, axis=1)
    k = np.array([100., 100., 100., 100.])
    init_pops = np.array([5., 5., 5., 5.])
    trans_pops = np.array([8., 8., 8., 8.])
    # data, low, high = map_data(init_pops, trans_pops, k, 1000, 1000, overlay, 10)
    # dmat = map_dmat(data, low, high)
    dmat = map_data(init_pops, trans_pops, k, 100., 1000., overlay, 5.)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(dmat, cmap=plt.cm.Blues)
    plt.show(heatmap)
    np.savetxt("LVD_map_dmat.txt", dmat, fmt='%.5f')