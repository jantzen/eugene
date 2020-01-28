# file: test_dynamical_distance.py

import eugene as eu
from eugene.src.virtual_sys.LotkaVolterraSND import LotkaVolterraSND
from eugene.src.data_prep.initial_conditions import choose_untrans_trans
from eugene.src.dynamical_distance import distance_matrix
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def data_sample(interval, samples, system):
    curve = np.asarray(system._x).reshape(1, -1)
    for i in range(samples):
        system.update_x(interval)
        tmp = np.asarray(system._x).reshape(1, -1)
        curve = np.vstack((curve, tmp))

    return curve.T
 
def test_distance_matrix():

    ########## 4D
    samples=100 
    reps=100 
    choose_alpha=0.5
    choose_beta=0.2 

    # generate data for two systems with LVDSim
    # Set params
    r1 = [1.1, 1.1]
    r2 = [2.2, 2.2]
    r3 = [1.1, 2.2]
    r4 = [1.1, 1.1]
    k1 = [100., 100.]
    k2 = [100., 100.]
    k3 = [100., 100.]
    k4 = [150., 150.]
    alpha = [[1., 0.5], [1.5, 1.]]
    sigma = [0.1,0.1]
    tmax = 2.

    # build the systems
    sys1 = []
    sys2 = []
    sys3 = []
    sys4 = []
    for i in range(reps):
        x = np.random.rand(8)*7. + 10.
        init_x1 = x[:2]
        init_x2 = x[2:4]
        init_x3 = x[4:6]
        init_x4 = x[-2:]
        sys1.append(LotkaVolterraSND(r1, k1, alpha, sigma, init_x1))
        sys2.append(LotkaVolterraSND(r2, k2, alpha, sigma, init_x2))
        sys3.append(LotkaVolterraSND(r3, k3, alpha, sigma, init_x3))
        sys4.append(LotkaVolterraSND(r4, k4, alpha, sigma, init_x4))

    # collect data
    cpus = max(multiprocessing.cpu_count() - 2, 1)
    interval = tmax / samples

    data1 = []
    data2 = []
    data3 = []
    data4 = []

    out1 = Parallel(n_jobs=cpus, verbose=25)(delayed(data_sample)(interval,
        samples, system) for system in sys1)
    for seg in out1:
        data1.append(seg)
    out2 = Parallel(n_jobs=cpus, verbose=25)(delayed(data_sample)(interval,
        samples, system) for system in sys2)
    for seg in out2:
        data2.append(seg)
    out3 = Parallel(n_jobs=cpus, verbose=25)(delayed(data_sample)(interval,
        samples, system) for system in sys3)
    for seg in out3:
        data3.append(seg)
    out4 = Parallel(n_jobs=cpus, verbose=25)(delayed(data_sample)(interval,
        samples, system) for system in sys4)
    for seg in out4:
        data4.append(seg)

    # select segments
    untrans, trans = choose_untrans_trans([data1, data2, data3, data4], 10,
            alpha=choose_alpha, beta=choose_beta)

    # build distance matrix
    print("Computing distance matrix in parallel.\n")
    d_parallel = distance_matrix(untrans, trans, 10)
    print("Computing distance matrix in serial.\n")
    d_linear = distance_matrix(untrans, trans, 10, parallel_compute=False)

    assert np.all(d_parallel == d_linear)
    assert d_parallel[0,1] < d_parallel[0,2]
    assert d_parallel[0,3] < d_parallel[0,2]
    assert d_parallel[1,3] < d_parallel[1,2]


    ####################
    # test 1-D

    data1_1d = []
    data2_1d = []
    data3_1d = []
    data4_1d = []

    for seg in data1:
        data1_1d.append(np.mean(seg, axis=0))
    for seg in data2:
        data2_1d.append(np.mean(seg, axis=0))
    for seg in data3:
        data3_1d.append(np.mean(seg, axis=0))
    for seg in data4:
        data4_1d.append(np.mean(seg, axis=0))

    # select segments
    untrans, trans = choose_untrans_trans([data1_1d, data2_1d, data3_1d,
        data4_1d], 10, alpha=choose_alpha, beta=choose_beta)

    # build distance matrix
    print("Computing 1-D  distance matrix in parallel.\n")
    d_parallel = distance_matrix(untrans, trans, 10)
    print("Computing 1-D distance matrix in serial.\n")
    d_linear = distance_matrix(untrans, trans, 10, parallel_compute=False)

    assert np.all(d_parallel == d_linear)
