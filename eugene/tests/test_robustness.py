# test_robustness.py
# No, this shouldn't be a test. I agree.

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from eugene.src.tools.LVDSim import *
from eugene.src.tools.StochasticTools import *
import unittest
from os.path import join, isfile
import sys
import copy
# a hack to import from AutomatedDiscovery and VolcanoData
sys.path.insert(0, '../../../AutomatedDiscovery/DADS2018/')
import approximate_kinds
import clipping


# TODO: create toolbox method to take 2 data sets and produce distances.
def parallel_distance(i, j, data, reps=True, clip=True):
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


def parallel_distance_rescan(i, j, data, params, max_time, num_time, overlay, stochastic_reps=None):
    # baseline is [sys1_untrans, sys2_untrans]
    untrans = [data[i][0], data[j][0]]
    # segment is [sys1_trans, sys2_trans]
    trans = [data[i][1], data[j][1]]

    if not stochastic_reps:
        # baseline is [sys1_untrans, sys2_untrans]
        untrans = [[np.array(data[i][0]).T], [np.array(data[j][0]).T]]
        # segment is [sys1_trans, sys2_trans]
        trans = [[np.array(data[i][1]).T], [np.array(data[j][1]).T]]

    d = _distance_rescan(untrans, trans, 5, params, max_time, num_time, overlay,
                         stochastic_reps)

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


def _distance_rescan(untrans, trans, min_length, params, max_time, num_time, overlay, stochastic_reps=None):
    ii = 0
    jj = 1

    # untrans[system, rep, species, t] or untrans[system, rep, t]
    # untrans[ii][rep, species, t] or untrans[ii][rep, t]
    print(np.shape(untrans[ii]))
    condition1 = untrans[ii]
    condition2 = untrans[jj]

    # clip the data
    tmp_untrans1, tmp_untrans2 = clipping.clip_segments(condition1, condition2,
            min_length)
    cut_len1 = np.shape(tmp_untrans1[0])[-1]
    cut_len2 = np.shape(tmp_untrans2[0])[-1]

    # double check
    a = np.shape(condition1[0])[-1]
    b = np.shape(condition2[0])[-1]

    assert cut_len1 == a or cut_len2 == b

    # instead of making one series shorter, re-run the system in new time bounds
    if cut_len1 < cut_len2:
        # tmp_trans2 is fine
        tmp_trans2 = copy.deepcopy(trans[jj])
        # tmp_trans1 and tmp_untrans1 need to be re-run
        param = params[0]
        new_t = (cut_len1/num_time)*max_time
        new_data = simData([param], new_t, num_time, overlay,
                           stochastic_reps=stochastic_reps, range_cover=False)
        if not stochastic_reps:
            tmp_untrans1 = [new_data[0][0].T]
            tmp_trans1 = [new_data[0][1].T]
        else:
            tmp_untrans1 = new_data[0][0]
            tmp_trans1 = new_data[0][1]
    elif cut_len1 > cut_len2:
        # tmp_trans1 is fine
        tmp_trans1 = copy.deepcopy(trans[ii])
        # tmp_trans2 and tmp_untrans2 need to be re-run
        param = params[1]
        new_t = (cut_len2/num_time)*max_time
        new_data = simData([param], new_t, num_time, overlay,
                           stochastic_reps=stochastic_reps, range_cover=False)
        if not stochastic_reps:
            tmp_untrans2 = [new_data[0][0].T]
            tmp_trans2 = [new_data[0][1].T]
        else:
            tmp_untrans2 = new_data[0][0]
            tmp_trans2 = new_data[0][1]
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


def stochastic_generation_loop_times(test_name, params, max_time, n_time,
                                     overlay, reps, clip=True):
    data_name = test_name + str(n_time)
    data_path = join("test_data", data_name + ".npy")
    data = None
    if isfile(data_path):
        data = np.load(data_path)
    else:
        data = simData(params, max_time, n_time, overlay,
                       stochastic_reps=reps, range_cover=False)
        np.save(data_path, data)
    test = TestRobustness()
    dist = test.pair_dist(data, clip=clip)
    return [dist, n_time]


def stochastic_generation_loop_reps(test_name, params, max_time, n_time,
                                    overlay, reps, clip=True, overlay_flag=False):
    data_name = test_name + str(reps)
    if overlay_flag:
        data_path = join("test_data", data_name + ".npy")
    else:
        data_path = join("test_data", data_name + ".npy")
    data = None
    if isfile(data_path):
        data = np.load(data_path)
    else:
        data = simData(params, max_time, n_time, overlay,
                       stochastic_reps=reps, range_cover=False)
        np.save(data_path, data)
    test = TestRobustness()
    dist = test.pair_dist(data, clip=clip)
    return [dist, reps]


def stochastic_generation_loop_sigma(test_name, params, max_time, n_time,
                                     overlay, reps, clip=True):
    data_name = test_name + str(params[0][3][0])
    data_path = join("test_data", data_name + ".npy")
    data = None
    if isfile(data_path):
        data = np.load(data_path)
    else:
        data = simData(params, max_time, n_time, overlay,
                       stochastic_reps=reps, range_cover=False)
        np.save(data_path, data)
    test = TestRobustness()
    dist = test.pair_dist(data, clip=clip)
    return [dist, params[0][3][0]]


def setup_params(same=True, stochastic=False, sigma = [0.1, 0.1]):
    r1 = np.array([1., 2.])
    k1 = np.array([100., 100.])

    alpha1 = np.array([[1., 0.5], [0.7, 1.]])
    alpha2 = alpha1

    if same:
        r2 = r1 * 1.5
        k2 = k1
    else:
        r2 = r1
        k2 = np.array([150., 150.])

    init1 = np.array([5., 5.])
    init2 = init1

    init_trans1 = np.array([8., 8.])
    init_trans2 = init_trans1

    if stochastic:
        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]
    else:
        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

    return [params1, params2]


def mean_overlay(x):
    return np.mean(x, axis=1)


def no_overlay(x):
    return x


class TestRobustness(unittest.TestCase):


    def pair_dist(self, data, free_cores=2, reps=True, clip=True):
        cpus = max(cpu_count() - free_cores, 1)
        s = len(data)
        distances = Parallel(n_jobs=cpus, verbose=5)(
            delayed(parallel_distance)(i, j, data, reps, clip) for i in range(s) for j
            in range(i, s))

        dmat = np.zeros((s, s))
        for cell in distances:
            #        print cell
            dmat[cell[0], cell[1]] = dmat[cell[1], cell[0]] = cell[2]
        return dmat[0][1]


    def pair_dist_rescan(self, data, params, max_time, num_time, overlay, stochastic_reps=None, free_cores=2):
        cpus = max(cpu_count() - free_cores, 1)
        s = len(data)
        distances = Parallel(n_jobs=cpus, verbose=5)(
            delayed(parallel_distance_rescan)(i, j, data, params, max_time, num_time, overlay, stochastic_reps) for i in range(s) for j
            in range(i, s))

        dmat = np.zeros((s, s))
        for cell in distances:
            #        print cell
            dmat[cell[0], cell[1]] = dmat[cell[1], cell[0]] = cell[2]
        return dmat[0][1]


    @unittest.skip
    def test_into_chaos_growth(self):
        test_name = "test_into_chaos_growth"
        print(test_name)

        r1 = np.array([1.7741, 1.0971, 1.5466, 4.4116])
        k1 = np.array([100., 100., 100., 100.])
        alpha1 = np.array([[1., 2.419, 2.248, 0.0023],
                           [0.001, 1., 0.001, 1.3142],
                           [2.3818, 0.001, 1., 0.4744],
                           [1.21, 0.5244, 0.001, 1.]])
        init1 = np.array([5., 5., 5., 5.])
        init_trans1 = np.array([8., 8., 8., 8.])

        r2 = r1
        k2 = k1
        alpha2 = alpha1
        init2 = init1
        init_trans2 = init_trans1

        alpha3 = np.array([[1., 2.5, 2.5, 0.1],
                            [0.1, 1., 0.1, 1.5],
                            [2.5, 0.1, 1., 0.5],
                            [1.5, 0.5, 0.1, 1.]])

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        params3 = [r1, k1, alpha3, init1, init_trans1]
        params4 = [r2, k2, alpha3, init2, init_trans2]

        dists = []
        dists2 = []
        lens = []
        for x in trange(101):
            n = x/50
            lens.append(n)
            data_name = test_name + "-" + str(n)
            data_path = join("test_data", data_name + ".npy")

            params2[0] = n*r1
            params = [params1, params2]

            params4[0] = n*r1
            params0 = [params3, params4]
            data = None
            data2 = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 100., 200, no_overlay, range_cover=False)
                data2 = simData(params0, 100., 200, no_overlay, range_cover=False)
                # np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=True)
            dist2 = self.pair_dist(data2, reps=False, clip=True)
            dists.append(dist)
            dists2.append(dist2)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'ro', label='chaotic base')
        ax.plot(lens, dists2, 'bo', label='non-chaotic base')
        ax.legend()
        ax.set(xlabel='scale of growth rate', ylabel='distance',
               title='Two Systems with Scaled Growth Rate through Chaos')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    # @unittest.skip
    def test_into_chaos_interaction(self):
        test_name = "test_into_chaos_interaction"
        print(test_name)

        r1 = np.array([1.7741, 1.0971, 1.5466, 4.4116])
        k1 = np.array([100., 100., 100., 100.])
        alpha1 = np.array([[1., 2.419, 2.248, 0.0023],
                           [0.001, 1., 0.001, 1.3142],
                           [2.3818, 0.001, 1., 0.4744],
                           [1.21, 0.5244, 0.001, 1.]])
        init1 = np.array([5., 5., 5., 5.])
        init_trans1 = np.array([8., 8., 8., 8.])

        r2 = r1
        k2 = k1
        alpha2 = alpha1
        init2 = init1
        init_trans2 = init_trans1

        alpha_s = np.array([[1., 0.9, 1., 1.],
                           [1., 1., 1., 1.],
                           [1., 1., 1., 1.],
                           [1., 1., 1., 1.]])

        alpha3 = np.array([[1., 2.5, 2.5, 0.1],
                            [0.1, 1., 0.1, 1.5],
                            [2.5, 0.1, 1., 0.5],
                            [1.5, 0.5, 0.1, 1.]])

        # alpha3 = alpha1 * np.power(alpha_s, 1.2)
        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        params3 = [r1, k1, alpha3, init1, init_trans1]
        params4 = [r2, k2, alpha3, init2, init_trans2]

        dists = []
        dists2 = []
        lens = []
        for x in trange(101):
            n = (x/50 - 1)*0.1
            lens.append(n)
            data_name = test_name + "-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            tmp_val = 2.419 + n
            tmp_a = np.array([[1., tmp_val, 2.248, 0.0023],
                               [0.001, 1., 0.001, 1.3142],
                               [2.3818, 0.001, 1., 0.4744],
                               [1.21, 0.5244, 0.001, 1.]])

            temp_a = alpha1 * np.power(alpha_s, n)
            params2[2] = tmp_a
            # lens.append(tmp_val)
            params = [params1, params2]

            tmp_val2 = 2.5 + n
            tmp_a2 = np.array([[1., tmp_val2, 2.5, 0.1],
                               [0.1, 1., 0.1, 1.5],
                               [2.5, 0.1, 1., 0.5],
                               [1.5, 0.5, 0.1, 1.]])
            params4[2] = tmp_a2
            params0 = [params3, params2]
            data = None
            data2 = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 100., 200, no_overlay, range_cover=False)
                data2 = simData(params0, 100., 200, no_overlay, range_cover=False)
                # np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=True)
            dist2 = self.pair_dist(data2, reps=False, clip=True)
            dists.append(dist)
            dists2.append(dist2)

        print(dists)
        print(dists2)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'ro', label='chaotic base')
        ax.plot(lens, dists2, 'bo', label='non-chaotic base')
        ax.legend()
        ax.set(xlabel='change of interaction term (0,1)', ylabel='distance',
               title='Two Systems with Different Interaction Term through Chaos')
        # plt.show()
        plt.savefig(test_name + "-long.pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_diff(self):
        print("test_same_more_diff")

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5, 5])
        init2 = init1

        init_trans1 = np.array([8, 8])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]
        params3 = [r1, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        dists2 = []
        lens = []
        for x in trange(10):
            n = x + 1
            lens.append(n)
            data_name = "test_same_more_diff-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            params2[1] = k2 * n
            params3[0] = r1 * n
            data = None
            if isfile(data_path):
                # data = np.load(data_path)
                pass
            else:
                data = simData([params1, params2], 15., 50, no_overlay, range_cover=False)
                data2 = simData([params1, params3], 15., 50, no_overlay, range_cover=False)
                # np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=True)
            dist2 = self.pair_dist(data2, reps=False, clip=True)
            dists.append(dist)
            dists2.append(dist2)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo', label='parameter = capacities')
        ax.plot(lens, dists2, 'ro', label='parameter = growth rates')
        ax.legend()
        ax.set(xlabel='x * parameter', ylabel='distance',
               title='Distance Between Two Systems')
        # plt.show()
        plt.savefig(data_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_into_chaos_interaction_noise(self):
        test_name = "test_into_chaos_interaction_noise"
        print(test_name)

        r1 = np.array([1.7741, 1.0971, 1.5466, 4.4116])
        k1 = np.array([100., 100., 100., 100.])
        alpha1 = np.array([[1., 2.419, 2.248, 0.0023],
                           [0.001, 1., 0.001, 1.3142],
                           [2.3818, 0.001, 1., 0.4744],
                           [1.21, 0.5244, 0.001, 1.]])
        init1 = np.array([5., 5., 5., 5.])
        init_trans1 = np.array([8., 8., 8., 8.])

        r2 = r1
        k2 = k1
        alpha2 = alpha1
        init2 = init1
        init_trans2 = init_trans1

        alpha_s = np.array([[1., 0.9, 1., 1.],
                            [1., 1., 1., 1.],
                            [1., 1., 1., 1.],
                            [1., 1., 1., 1.]])

        alpha3 = np.array([[1., 2.5, 2.5, 0.1],
                           [0.1, 1., 0.1, 1.5],
                           [2.5, 0.1, 1., 0.5],
                           [1.5, 0.5, 0.1, 1.]])

        # alpha3 = alpha1 * np.power(alpha_s, 1.2)
        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        params3 = [r1, k1, alpha3, init1, init_trans1]
        params4 = [r2, k2, alpha3, init2, init_trans2]

        dists = []
        dists2 = []
        lens = []
        for x in trange(101):
            n = (x / 50 - 1) * 0.1
            lens.append(n)
            data_name = test_name + "-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            tmp_val = 2.419 + n
            tmp_a = np.array([[1., tmp_val, 2.248, 0.0023],
                              [0.001, 1., 0.001, 1.3142],
                              [2.3818, 0.001, 1., 0.4744],
                              [1.21, 0.5244, 0.001, 1.]])

            temp_a = alpha1 * np.power(alpha_s, n)
            params2[2] = tmp_a
            # lens.append(tmp_val)
            params = [params1, params2]

            tmp_val2 = 2.5 + n
            tmp_a2 = np.array([[1., tmp_val2, 2.5, 0.1],
                               [0.1, 1., 0.1, 1.5],
                               [2.5, 0.1, 1., 0.5],
                               [1.5, 0.5, 0.1, 1.]])
            params4[2] = tmp_a2
            params0 = [params3, params2]
            data = None
            data2 = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 100., 200, no_overlay, range_cover=False)
                data2 = simData(params0, 100., 200, no_overlay,
                                range_cover=False)
                # np.save(data_path, data)
            noisy_data = []
            for tup in data:
                ntup = []
                for curve in tup:
                    ncurve = []
                    for point in curve:
                        ncurve.append(point + np.random.normal(0., 2))
                    ntup.append(ncurve)
                noisy_data.append(ntup)

            data = np.array(noisy_data)

            noisy_data2 = []
            for tup in data2:
                ntup = []
                for curve in tup:
                    ncurve = []
                    for point in curve:
                        ncurve.append(point + np.random.normal(0., 2))
                    ntup.append(ncurve)
                noisy_data2.append(ntup)

            data2 = np.array(noisy_data2)
            dist = self.pair_dist(data, reps=False, clip=True)
            dist2 = self.pair_dist(data2, reps=False, clip=True)
            dists.append(dist)
            dists2.append(dist2)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'ro', label='chaotic base')
        ax.plot(lens, dists2, 'bo', label='non-chaotic base')
        ax.legend()
        ax.set(xlabel='change of interaction term (0,1)', ylabel='distance',
               title='Two Systems with Different Interaction Term through Chaos')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_cap_more_linear(self):
        test_name = "test_same_cap_more_linear"
        print(test_name)

        params = setup_params()
        params1 = params[0]
        params1.append(0.)
        params2 = params[1]
        params2.append(0.)

        dists = []
        lens = []
        for x in trange(10):
            n = x/5
            # n = np.tan(np.power(x, 2))
            lens.append(n)
            data_name = test_name + "-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            params1[-1] = 1.
            params2[-1] = n
            params = [params1, params2]
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simDataLin(params, 10., 20, no_overlay)
                # np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=True)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='linearity factor', ylabel='distance',
               title='Two Systems with Same Capacity and Different Linearity')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_cap_more_2nd_order(self):
        test_name = "test_same_cap_more_2nd_order"
        print(test_name)

        init_y = [0., 0.]
        params = setup_params()
        params1 = params[0]
        params1.append(init_y)
        params1.append(0.)
        params2 = params[1]
        params2.append(init_y)
        params2.append(0.)

        dists = []
        lens = []
        for x in trange(10):
            n = np.power(10, x)
            # n = np.tan(np.power(x, 2))
            lens.append(n)
            data_name = test_name + "-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            params1[-1] = 1.
            params2[-1] = n
            params = [params1, params2]
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData2OD(params, 10., 20, no_overlay,
                               range_cover=False)
                # np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=True)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='order factor', ylabel='distance', xscale='log',
               title='Two Systems with Same Capacity and Different Effective Order')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data(self):
        print ("test_same_more_data")

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5, 5])
        init2 = init1

        init_trans1 = np.array([8, 8])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_data-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, overlay, range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of the Same Kind')
        # plt.show()
        plt.savefig(data_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data_noclip(self):
        test_name = "test_same_more_data_noclip"
        print(test_name)

        params = setup_params()
        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_data-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of the Same Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data_noisy(self):
        print("test_same_more_data_noisy")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5, 5])
        init2 = init1

        init_trans1 = np.array([8, 8])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        stdev = 1.0
        for x in trange(9):
            n = 10.0 * np.power(2, x + 1)
            lens.append(n)
            data_name = "test_same_more_data_noisy-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, overlay, range_cover=False)
                noisy_data = []
                for tup in data:
                    ntup = []
                    for curve in tup:
                        ncurve = []
                        for point in curve:
                            ncurve.append(point + np.random.normal(0., stdev))
                        ntup.append(ncurve)
                    noisy_data.append(ntup)

                data = np.array(noisy_data)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of the Same Kind with Noise')
        # plt.show()
        plt.savefig(data_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data_noisy_noclip(self):
        test_name = "test_same_more_data_noisy_noclip"
        print(test_name)

        params = setup_params()

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        stdev = 1.0
        for x in trange(9):
            n = 10.0 * np.power(2, x + 1)
            lens.append(n)
            data_name = "test_same_more_data_noisy-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay, range_cover=False)
                noisy_data = []
                for tup in data:
                    ntup = []
                    for curve in tup:
                        ncurve = []
                        for point in curve:
                            ncurve.append(point + np.random.normal(0., stdev))
                        ntup.append(ncurve)
                    noisy_data.append(ntup)

                data = np.array(noisy_data)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of the Same Kind with Noise, whithout Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data_stochastic(self):
        print ("test_same_more_data_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []

        cpus = max(cpu_count() - 2, 1)
        test_name = "test_same_more_data_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(delayed(stochastic_generation_loop_times)(test_name, [params1, params2], 5., 10.0 * np.power(2, x + 1), mean_overlay, 10) for x in trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]
        # for x in trange(9):
        #     n = 10.0 * np.power(2, x+1)
        #     lens.append(n)
        #     data_name = "test_same_more_data_stochastic-" + str(n)
        #     data_path = join("test_data", data_name + ".npy")
        #     data = None
        #     if isfile(data_path):
        #         data = np.load(data_path)
        #     else:
        #         data = simData([params1, params2], 5., n, overlay, stochastic_reps=10, range_cover=False)
        #         np.save(data_path, data)
        #     dist = self.pair_dist(data)
        #     dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Stochastic Systems of the Same Kind')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_data_stochastic_noclip(self):
        test_name = "test_same_more_data_stochastic_noclip"
        print(test_name)

        params = setup_params(stochastic=True)

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []

        cpus = max(cpu_count() - 2, 1)
        data_name = "test_same_more_data_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_times)(data_name,
                                                      params, 5.,
                                                      10.0 * np.power(2, x + 1),
                                                      mean_overlay, 10, clip=False) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]
        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Stochastic Systems of the Same Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_reps_stochastic(self):
        print ("test_same_more_reps_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=-2)
        # overlay = lambda x: x

        cpus = max(cpu_count() - 1, 1)
        test_name = "test_same_more_reps_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_reps)(test_name,
                                                     [params1, params2],
                                                     5., 500, no_overlay,
                                                     x+1) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]

        # dists = []
        # for x in trange(10):
        #     data = simData([params1, params2], 5., 500, overlay, stochastic_reps=x+1, range_cover=False)
        #     dist = self.pair_dist(data)
        #     dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='stochastic replications', ylabel='distance',
               title='Distance Between Two Stochastic Systems of the Same Kind')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_same_more_reps_stochastic_noclip(self):
        test_name = "test_same_more_reps_stochastic_noclip"
        print(test_name)

        params = setup_params(stochastic=True)

        overlay = lambda x: np.mean(x, axis=-2)
        # overlay = lambda x: x

        cpus = max(cpu_count() - 1, 1)
        data_name = "test_same_more_reps_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_reps)(data_name,
                                                     params,
                                                     5., 500, no_overlay,
                                                     x + 1, clip=False) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='stochastic replications', ylabel='distance',
               title='Distance Between Two Stochastic Systems of the Same Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_both_more_noise_stochastic(self):
        print ("test_both_more_noise_stochastic")
        test_name = "test_both_more_noise_stochastic"

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])
        k3 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        cpus = max(cpu_count() - 2, 1)
        test_name1 = "test_both_same_more_noise_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(test_name1,
                                                     [[r1, k1, alpha1,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init1, init_trans1],
                                                      [r2, k2, alpha2,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init2, init_trans2]],
                                                     15., 50, no_overlay,
                                                     10) for x in
            trange(10))
        test_name2 = "test_both_diff_more_noise_stochastic-"
        dist_times2 = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(test_name2,
                                                     [[r1, k1, alpha1,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init1, init_trans1],
                                                      [r2, k3, alpha2,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init2, init_trans2]],
                                                     15., 50, no_overlay,
                                                     10) for x in
            trange(10))
        test_name3 = "test_both_same_more_noise_less_info_stochastic-"
        dist_times3 = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(test_name3,
                                                     [[r1, k1, alpha1,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init1, init_trans1],
                                                      [r2, k2, alpha2,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init2, init_trans2]],
                                                     15., 50, mean_overlay,
                                                     10) for x in
            trange(10))
        test_name4 = "test_both_diff_more_noise_less_info_stochastic-"
        dist_times4 = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(test_name4,
                                                     [[r1, k1, alpha1,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init1, init_trans1],
                                                      [r2, k3, alpha2,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init2, init_trans2]],
                                                     15., 50, mean_overlay,
                                                     10) for x in
            trange(10))
        dists = np.array(dist_times)[:, 0]
        dists2 = np.array(dist_times2)[:, 0]
        dists3 = np.array(dist_times3)[:, 0]
        dists4 = np.array(dist_times4)[:, 0]
        sigma = np.array(dist_times)[:, 1]
        # for x in trange(10):
        #     sigma = [0.1, 0.1]*(x + 1)
        #     params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        #     params2 = [r2, k2, alpha2, sigma, init2, init_trans2]
        #     data = simData([params1, params2], 5., 500, overlay, stochastic_reps=10, range_cover=False)
        #     dist = self.pair_dist(data)
        #     dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(sigma, dists, 'bo', label='same kind, SSD')
        ax.plot(sigma, dists2, 'ro', label='different kind, SSD')
        ax.plot(sigma, dists3, 'b+', label='same kind, non-SSD')
        ax.plot(sigma, dists4, 'r+', label='different kind, non-SSD')
        ax.set(xlabel='sigma', ylabel='distance',
               title='Distance Between Two Stochastic Systems')
        ax.legend()
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() < dists2.var())


    @unittest.skip
    def test_same_more_noise_stochastic(self):
        print ("test_same_more_noise_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1 * 1.5

        k1 = np.array([100., 100.])
        k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        cpus = max(cpu_count() - 2, 1)
        test_name = "test_same_more_noise_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(test_name,
                                                     [[r1, k1, alpha1,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init1, init_trans1],
                                                      [r2, k2, alpha2,
                                                       np.array([0.1, 0.1])*(x + 1),
                                                       init2, init_trans2]],
                                                     5., 500, mean_overlay,
                                                     10) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        sigma = np.array(dist_times)[:, 1]
        # for x in trange(10):
        #     sigma = [0.1, 0.1]*(x + 1)
        #     params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        #     params2 = [r2, k2, alpha2, sigma, init2, init_trans2]
        #     data = simData([params1, params2], 5., 500, overlay, stochastic_reps=10, range_cover=False)
        #     dist = self.pair_dist(data)
        #     dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(sigma, dists, 'bo')
        ax.set(xlabel='sigma', ylabel='distance',
               title='Distance Between Two Stochastic Systems of the Same Kind')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() < dists2.var())


    @unittest.skip
    def test_same_more_noise_stochastic_noclip(self):
        test_name = "test_same_more_noise_stochastic_noclip"
        print(test_name)

        dists = []
        cpus = max(cpu_count() - 2, 1)
        data_name = "test_same_more_noise_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(data_name,
                                                      setup_params(
                                                          stochastic=True,
                                                          sigma=np.array(
                                                              [0.1, 0.1]) * (
                                                                        x + 1)),
                                                      5., 500, mean_overlay,
                                                      10, clip=False) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        sigma = np.array(dist_times)[:, 1]

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(sigma, dists, 'bo')
        ax.set(xlabel='sigma', ylabel='distance',
               title='Distance Between Two Stochastic Systems of the Same Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() < dists2.var())


    @unittest.skip
    def test_diff_more_data(self):
        print ("test_diff_more_data")

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x+1)
            lens.append(n)
            data_name = "test_diff_more_data-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists.append(dist)

        print (dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of Different Kind')
        # plt.show()
        plt.savefig(data_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data_noclip(self):
        test_name = "test_diff_more_data_noclip"
        print(test_name)

        params = setup_params(same=False)

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x + 1)
            lens.append(n)
            data_name = "test_diff_more_data-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of Different Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data_noisy(self):
        print("test_diff_more_data_noisy")

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        stdev = 1.0
        for x in trange(9):
            n = 10.0 * np.power(2, x + 1)
            lens.append(n)
            data_name = "test_diff_more_data_noisy-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, overlay,
                               range_cover=False)
                noisy_data = []
                for tup in data:
                    ntup = []
                    for curve in tup:
                        ncurve = []
                        for point in curve:
                            ncurve.append(point + np.random.normal(0., stdev))
                        ntup.append(ncurve)
                    noisy_data.append(ntup)

                data = np.array(noisy_data)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of Different Kind with Noise')
        # plt.show()
        plt.savefig(data_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data_noisy_noclip(self):
        test_name = "test_diff_more_data_noisy_noclip"
        print(test_name)

        params = setup_params(same=False)

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        stdev = 1.0
        for x in trange(9):
            n = 10.0 * np.power(2, x + 1)
            lens.append(n)
            data_name = "test_diff_more_data_noisy-" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay,
                               range_cover=False)
                noisy_data = []
                for tup in data:
                    ntup = []
                    for curve in tup:
                        ncurve = []
                        for point in curve:
                            ncurve.append(point + np.random.normal(0., stdev))
                        ntup.append(ncurve)
                    noisy_data.append(ntup)

                data = np.array(noisy_data)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists.append(dist)

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of Different Kind with Noise, without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data_stochastic(self):
        print ("test_diff_more_data_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []

        cpus = max(cpu_count() - 2, 1)
        test_name = "test_diff_more_data_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_times)(test_name,
                                                      [params1, params2], 5.,
                                                      10.0 * np.power(2, x + 1),
                                                      mean_overlay, 10) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]
        # for x in trange(10):
        #     n = 10.0 * np.power(2, x+1)
        #     lens.append(n)
        #     data = simData([params1, params2], 5., n, overlay, stochastic_reps=10, range_cover=False)
        #     dist = self.pair_dist(data)
        #     dists.append(dist)

        print (dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Stochastic Systems of Different Kind')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_data_stochastic_noclip(self):
        test_name = "test_diff_more_data_stochastic_noclip"
        print(test_name)

        params = setup_params(same=False, stochastic=True)

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []

        cpus = max(cpu_count() - 2, 1)
        data_name = "test_diff_more_data_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_times)(data_name,
                                                      params, 5.,
                                                      10.0 * np.power(2, x + 1),
                                                      mean_overlay, 10, clip=False) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance between Two Stochastic Systems of Different Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_reps_stochastic(self):
        print ("test_diff_more_reps_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        cpus = max(cpu_count() - 2, 1)
        test_name = "test_diff_more_reps_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_reps)(test_name,
                                                     [params1, params2],
                                                     5., 500, mean_overlay,
                                                     x + 1) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]

        # dists = []
        # for x in trange(10):
        #     data = simData([params1, params2], 5., 500, overlay, stochastic_reps=x+1, range_cover=False)
        #     dist = self.pair_dist(data)
        #     dists.append(dist)

        print (dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='stochastic replications', ylabel='distance',
               title='Distance Between Two Stochastic Systems of Different Kind')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_reps_stochastic_noclip(self):
        test_name = "test_diff_more_reps_stochastic_noclip"
        print(test_name)

        params = setup_params(same=False, stochastic=True)

        overlay = lambda x: np.mean(x, axis=1)

        cpus = max(cpu_count() - 2, 1)
        data_name = "test_diff_more_reps_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_reps)(data_name,
                                                     params,
                                                     5., 500, mean_overlay,
                                                     x + 1, clip=False) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        lens = np.array(dist_times)[:, 1]

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo')
        ax.set(xlabel='stochastic replications', ylabel='distance',
               title='Distance Between Two Stochastic Systems of Different Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() > dists2.var())


    @unittest.skip
    def test_diff_more_noise_stochastic(self):
        print ("test_diff_more_noise_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        cpus = max(cpu_count() - 2, 1)
        test_name = "test_diff_more_noise_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(test_name,
                                                     [[r1, k1, alpha1,
                                                       np.array([0.1, 0.1]) * (x + 1),
                                                       init1, init_trans1],
                                                      [r2, k2, alpha2,
                                                       np.array([0.1, 0.1]) * (x + 1),
                                                       init2, init_trans2]],
                                                     5., 500, mean_overlay,
                                                     10) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        sigma = np.array(dist_times)[:, 1]
        # for x in trange(10):
        #     sigma = [0.1, 0.1]*(x + 1)
        #     params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        #     params2 = [r2, k2, alpha2, sigma, init2, init_trans2]
        #     data = simData([params1, params2], 5., 500, overlay, stochastic_reps=10, range_cover=False)
        #     dist = self.pair_dist(data)
        #     dists.append(dist)

        print (dists)
        fig, ax = plt.subplots()
        ax.plot(sigma, dists, 'bo')
        ax.set(xlabel='sigma', ylabel='distance',
               title='Distance Between Two Stochastic Systems of Different Kind')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() < dists2.var())


    @unittest.skip
    def test_diff_more_noise_stochastic_noclip(self):
        test_name = "test_diff_more_noise_stochastic_noclip"
        print(test_name)

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        cpus = max(cpu_count() - 2, 1)
        data_name = "test_diff_more_noise_stochastic-"
        dist_times = Parallel(n_jobs=cpus)(
            delayed(stochastic_generation_loop_sigma)(data_name,
                                                      setup_params(
                                                          same=False,
                                                          stochastic=True,
                                                          sigma=np.array(
                                                              [0.1, 0.1]) * (
                                                                        x + 1)),
                                                      5., 500, mean_overlay,
                                                      10, clip=False) for x in
            trange(9))
        dists = np.array(dist_times)[:, 0]
        sigma = np.array(dist_times)[:, 1]

        print(dists)
        fig, ax = plt.subplots()
        ax.plot(sigma, dists, 'bo')
        ax.set(xlabel='sigma', ylabel='distance',
               title='Distance between Two Stochastic Systems of Different Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")
        dists1 = np.array(dists[0:3])
        dists2 = np.array(dists[-4:-1])
        self.assertTrue(dists1.var() < dists2.var())


    @unittest.skip
    def test_same_more_info(self):
        print ("test_same_more_info")
        # currently in commission as comparing more than one variable is...
        # easy!

        r1 = np.array([1., 2.])
        r2 = 1.5 * r1

        k1 = k2 = np.array([100., 100.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_info-" + "less_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_info-" + "more_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, less_overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists2.append(dist)

        print(dists2)

        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo', label='Less Info')
        ax.plot(lens, dists2, 'ro', label='More Info')
        ax.legend()
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of the Same Kind')
        # plt.show()
        plt.savefig(data_name + ".pdf")

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    @unittest.skip
    def test_same_more_info_noclip(self):
        test_name = "test_same_more_info_noclip"
        print(test_name)

        params = setup_params()

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_info-" + "less_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_info-" + "more_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, less_overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists2.append(dist)

        print(dists2)

        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo', label='Less Info')
        ax.plot(lens, dists2, 'ro', label='More Info')
        ax.legend()
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of the Same Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    @unittest.skip
    def test_same_more_info_rescan(self):
        test_name = "test_same_more_info_rescan"
        print(test_name)

        params = setup_params()

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_info-" + "less_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist_rescan(data, params, 5., n, overlay)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_same_more_info-" + "more_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, less_overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist_rescan(data, params, 5., n, less_overlay)
            dists2.append(dist)

        print(dists2)

        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo', label='Less Info')
        ax.plot(lens, dists2, 'ro', label='More Info')
        ax.legend()
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of the Same Kind with Rescan')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    @unittest.skip
    def test_diff_more_info(self):
        print ("test_diff_more_info")

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_diff_more_info-" + "less_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_diff_more_info-" + "more_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData([params1, params2], 5., n, less_overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False)
            dists2.append(dist)

        print(dists2)

        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo', label='Less Info')
        ax.plot(lens, dists2, 'ro', label='More Info')
        ax.legend()
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of Different Kind')
        # plt.show()
        plt.savefig(data_name + ".pdf")

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    @unittest.skip
    def test_diff_more_info_noclip(self):
        test_name = "test_diff_more_info_noclip"
        print(test_name)

        params = setup_params(same=False)

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_diff_more_info-" + "less_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_diff_more_info-" + "more_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, less_overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist(data, reps=False, clip=False)
            dists2.append(dist)

        print(dists2)

        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo', label='Less Info')
        ax.plot(lens, dists2, 'ro', label='More Info')
        ax.legend()
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of Different Kind without Clipping')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    @unittest.skip
    def test_diff_more_info_rescan(self):
        test_name = "test_diff_more_info_rescan"
        print(test_name)

        params = setup_params(same=False)

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_diff_more_info-" + "less_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist_rescan(data, params, 5., n, overlay)
            dists.append(dist)

        print(dists)

        less_overlay = lambda x: x

        dists2 = []
        lens = []
        for x in trange(9):
            n = 10.0 * np.power(2, x)
            lens.append(n)
            data_name = "test_diff_more_info-" + "more_info" + str(n)
            data_path = join("test_data", data_name + ".npy")
            data = None
            if isfile(data_path):
                data = np.load(data_path)
            else:
                data = simData(params, 5., n, less_overlay,
                               range_cover=False)
                np.save(data_path, data)
            dist = self.pair_dist_rescan(data, params, 5., n, less_overlay)
            dists2.append(dist)

        print(dists2)

        fig, ax = plt.subplots()
        ax.plot(lens, dists, 'bo', label='Less Info')
        ax.plot(lens, dists2, 'ro', label='More Info')
        ax.legend()
        ax.set(xlabel='samples', ylabel='distance', xscale='log',
               title='Distance Between Two Systems of Different Kind with Rescan')
        # plt.show()
        plt.savefig(test_name + ".pdf")

        self.assertTrue(np.array(dists).var() > np.array(dists2).var())


    @unittest.skip
    def test_diff_more_data_by_more_reps_stochastic(self):
        print("test_diff_more_data_by_more_reps_stochastic")

        sigma = [0.1, 0.1]

        r1 = np.array([1., 2.])
        r2 = r1

        k1 = np.array([100., 100.])
        k2 = np.array([150., 150.])

        alpha1 = np.array([[1., 0.5], [0.7, 1.]])
        alpha2 = alpha1

        init1 = np.array([5., 5.])
        init2 = init1

        init_trans1 = np.array([8., 8.])
        init_trans2 = init_trans1

        params1 = [r1, k1, alpha1, sigma, init1, init_trans1]
        params2 = [r2, k2, alpha2, sigma, init2, init_trans2]

        overlay = lambda x: np.mean(x, axis=1)

        dists = []
        variances = np.zeros((2, 2))
        values = [1, 10]
        for x in trange(2):
            n = 10.0 * np.power(2, values[x])
            inner_dists = []
            for y in trange(2):
                data = simData([params1, params2], 5., n, overlay,
                               stochastic_reps=values[y], range_cover=False)
                data = [data[0][0], data[1][0]]
                dist = self.pair_dist(data)
                # dist = energyDistanceMatrixParallel(data)
                inner_dists.append(dist)
                variances[x, y] = np.var(dist)
            dists.append(inner_dists)

        print(dists)
        print(variances)
        plt.imshow(variances, interpolation='nearest')
        self.assertTrue(variances[0,0] > variances[1, 1])


if __name__ == '__main__':
    unittest.main()
    plt.show()
