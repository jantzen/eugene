# LVD_map.py

from joblib import Parallel, delayed
from eugene.src.tools.LVDSim import simData
import multiprocessing
from eugene.src.tools.alphaBetaGrid import *
from eugene.src.auxiliary.probability import EnergyDistance
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import sys, os, pickle
import copy
# a hack to import from AutomatedDiscovery and VolcanoData
sys.path.insert(0, '../../../AutomatedDiscovery/DADS2018/')
import approximate_kinds
import clipping
# sys.path.insert(0, '../../../AutomatedDiscovery/VolcanoData/')
# from eugene_scan import _distance


def map_data(init_pops, trans_pops, caps, max_time, num_times, depth):
    num_cores = multiprocessing.cpu_count() - 2
    list_of_points = lv_map_points(alpha_steps=depth, beta_steps=depth)

    # create a session folder

    params = Parallel(n_jobs=num_cores,verbose=5)(delayed(point_to_param)(point, init_pops, trans_pops, caps) for point in list_of_points)

    data = Parallel(n_jobs=num_cores,verbose=5)(delayed(simData)([param], max_time, num_times, overlaid, range_cover=False) for param in params)

    # data = simData(params, max_time, num_times, overlay, range_cover=False)
    new_data = []
    for c in data:
        new_data.append(c[0])

    new_data = np.array(new_data)
    s = len(new_data)
    distances = Parallel(n_jobs=num_cores,verbose=5)(delayed(parallel_distance)(i, j, new_data) for i in range(s) for j in range(i, s))

    dmat = np.zeros((s, s))
    for cell in distances:
        #        print cell
        dmat[cell[0], cell[1]] = dmat[cell[1], cell[0]] = cell[2]
    return dmat


def point_to_param(point, init_pops, trans_pops, caps, stochastic=False):
    param = []
    if stochastic:
        sigma = [0.1, 0.1, 0.1, 0.1]
        param = [point[0], caps, point[1], sigma, init_pops, trans_pops]
    else:
        param = [point[0], caps, point[1], init_pops, trans_pops]

    return param


def overlaid(x):
    return x


def parallel_distance(i, j, data, reps=False, clip=True):
    # baseline is [sys1_untrans, sys2_untrans]
    untrans = [data[i][0], data[j][0]]
    # segment is [sys1_trans, sys2_trans]
    trans = [data[i][1], data[j][1]]

    if not reps:
        # baseline is [sys1_untrans, sys2_untrans]
        untrans = [[np.array(data[i][0]).T], [np.array(data[j][0]).T]]
        # segment is [sys1_trans, sys2_trans]
        trans = [[np.array(data[i][1]).T], [np.array(data[j][1]).T]]

    d = _distance(untrans, trans, 5, clip)

    print('{},{}: d = {}.'.format(i, j, d))

    return [i, j, d]


def _distance(untrans, trans, min_length, clip=True):
    ii = 0
    jj = 1

    # untrans[system, rep, species, t] or untrans[system, rep, t]
    # untrans[ii][rep, species, t] or untrans[ii][rep, t]
    print(np.shape(untrans[ii]))
    condition1 = untrans[ii]
    condition2 = untrans[jj]

    if clip:
        # clip the data
        tmp_untrans1, tmp_untrans2 = clipping.clip_segments(condition1, condition2,
                min_length)
        cut_len1 = np.shape(tmp_untrans1[0])[-1]
        cut_len2 = np.shape(tmp_untrans2[0])[-1]
    else:
        tmp_untrans1 = condition1
        tmp_untrans2 = condition2
        cut_len1 = np.shape(condition1[0])[-1]
        cut_len2 = np.shape(condition2[0])[-1]

    # double check
    a = np.shape(condition1[0])[-1]
    b = np.shape(condition2[0])[-1]

    assert cut_len1 == a or cut_len2 == b

    # the tmp_trans1.append(segment[:cut_len1]) line won't work if segment is nd
    # so only take from the last axis
    if cut_len1 < cut_len2:
        tmp_trans2 = copy.deepcopy(trans[jj])
        tmp_trans1 = []
        for segment in trans[ii]:
            # tmp_trans1.append(segment[:cut_len1])
            tmp_trans1.append(np.take(segment, range(cut_len1), -1))
    elif cut_len1 > cut_len2:
        tmp_trans1 = copy.deepcopy(trans[ii])
        tmp_trans2 = []
        for segment in trans[jj]:
            # tmp_trans2.append(segment[:cut_len2])
            tmp_trans2.append(np.take(segment, range(cut_len2), -1))
    else:
        tmp_trans1 = copy.deepcopy(trans[ii])
        tmp_trans2 = copy.deepcopy(trans[jj])

    data1 = []
    data2 = []
    untrans1c = []
    untrans2c = []
    for index, seg in enumerate(tmp_untrans1):
        if seg.shape[-1] == tmp_trans1[index].shape[-1]:
            data1.append(np.vstack((seg, tmp_trans1[index])))
            untrans1c.append(seg)
        else:
            s = np.min([seg.shape[-1], tmp_trans1[index].shape[-1]])
            data1.append(np.vstack((seg[:,:s], tmp_trans1[index][:,:s])))
            untrans1c.append(seg[:,:s])
    data1 = np.concatenate(data1, axis=-1)
    untrans1c = np.concatenate(untrans1c)
    for index, seg in enumerate(tmp_untrans2):
        if seg.shape[-1] == tmp_trans2[index].shape[-1]:
            data2.append(np.vstack((seg, tmp_trans2[index])))
            untrans2c.append(seg)
        else:
            s = np.min([seg.shape[-1], tmp_trans2[index].shape[-1]])
            data2.append(np.vstack((seg[:,:s], tmp_trans2[index][:,:s])))
            untrans2c.append(seg[:,:s])
    data2 = np.concatenate(data2, axis=-1)
    untrans2c = np.concatenate(untrans2c)

    # data should be of shape (species*2, t*reps)
    print(np.shape(data1))
    dist = EnergyDistance(data1.T, data2.T, gpu=False)

    return dist

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
    dist_mat= map_data(init_pops, trans_pops, k, 4000., 4000., 5.)
    plt.imshow(dist_mat, cmap='hot', interpolation='nearest')
    np.savetxt(folder+"LVD_map_dmat.txt", dist_mat, fmt='%.5f')
    plt.show()
