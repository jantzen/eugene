# file: dynamical_distance.py

from eugene.src.data_prep.clipping import *
import copy
import multiprocessing
from joblib import Parallel, delayed


def distance_matrix_loop_func(ii, jj, untrans, trans, minimum_length, steps, return_partials,
        gpu):

    untrans1 = copy.deepcopy(untrans[ii])
    untrans2 = copy.deepcopy(untrans[jj])
    trans1 = copy.deepcopy(trans[ii])
    trans2 = copy.deepcopy(trans[jj])

    # clip the data
    tmp_untrans1, tmp_untrans2 = clip_segments(untrans1, untrans2,
            minimum_length, steps)
    tmp_trans1, tmp_trans2 = clip_to_match(tmp_untrans1, tmp_untrans2, trans1,
            trans2)

    data1, data2 = zip_curves(tmp_untrans1, tmp_untrans2, tmp_trans1,
            tmp_trans2)


    dist = EnergyDistance(data1.T, data2.T, gpu=gpu)

    if return_partials:
        for index, seg in enumerate(tmp_untrans1):
            if seg.shape[1] == tmp_trans1[index].shape[1]:
                untrans1c.append(seg)
            else:
                s = np.min([seg.shape[1], tmp_trans1[index].shape[1]])
                untrans1c.append(seg[:,:s])
        untrans1c = np.concatenate(untrans1c, axis=1)
        for index, seg in enumerate(tmp_untrans2):
            if seg.shape[1] == tmp_trans2[index].shape[1]:
                untrans2c.append(seg)
            else:
                s = np.min([seg.shape[1], tmp_trans2[index].shape[1]])
                untrans2c.append(seg[:,:s])
        untrans2c = np.concatenate(untrans2c, axis=1)

        pdist = EnergyDistance(untrans1c.T, untrans2c.T, gpu=gpu)
        return [ii, jj, dist, pdist]
    else:
        return [ii, jj, dist]


def distance_matrix(untrans, trans, minimum_length, steps=10, parallel_compute=True, free_cores=2,
        return_partials=False, gpu=False):

    # initialize the matrix
    d = np.zeros((len(untrans), len(untrans)))

    cpus = max(multiprocessing.cpu_count() - free_cores, 1)

    if parallel_compute:
        out = Parallel(n_jobs=cpus,
                verbose=5)(delayed(distance_matrix_loop_func)(ii,jj,untrans,trans,
                    minimum_length,steps,return_partials,gpu) 
                    for ii in range(len(untrans)) 
                    for jj in range(ii, len(untrans)))
    else:
        out = []
        for ii in range(len(untrans)):
            for jj in range(ii, len(untrans)):
                    out.append(distance_matrix_loop_func(ii,jj,untrans,trans,
                        minimum_length,steps,return_partials,gpu))

    for cell in out:
        d[cell[0], cell[1]] = d[cell[1], cell[0]] = cell[2]

    if return_partials:
        pd = np.zeros((len(untrans), len(untrans)))
        for cell in out:
            pd[cell[0], cell[1]] = pd[cell[1], cell[0]] = cell[3]
        return d, pd
    else:
        return d
 
