# file: metaparameters.py

import eugene as eu
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def cost(error_matrix):
    m = np.copy(error_matrix)
    m[np.where(m==2)] = 1
    m[np.where(m==3)] = 2
    return np.sum(m)

def tune_ic_selection(series, num_frags_range=None, reps_range =
        None, min_len=5, iterations=4):
    """ Sets optimal values for the following metaparameters:
        num_frags: number of fragments into which to split each timeseries
        reps: number of replicates to select from each pool

        Method:

    """

    # gather info about input
    min_series_len = series[0].shape[1] 
    for ts in series:
        tmp = ts.shape[1]
        if  tmp < min_series_len:
            min_series_len = tmp
    print("min_series_len = {}".format(min_series_len))

    # set up ranges
    if num_frags_range is None:
        lo = int(min_series_len / (20 * min_len)) # the factors of 2 and 20 
        hi = int(min_series_len / (2 * min_len))  # leave room for clipping
        num_frags_range = [lo, hi]
        print("num_frags_range = {}".format(num_frags_range))

#    if reps_range is None:
#        lo = int(max(num_frags_range[0] / 100, 1))
#        hi = int(max(num_frags_range[1] / 5, lo))
#        reps_range = [lo, hi]
#        print("reps_range = {}".format(reps_range))

    # enter loop
    min_cost = np.inf
    best_params = ()

#    for iters in range(iterations):
#
#        while ii < 4:
#            # for each parameter, choose 4 values at random
#            num_frags = np.random.randint(num_frags_range[0], num_frags_range[1] + 1, size=4) 
#            reps = np.random.randint(reps_range[0], reps_range[1] + 1, size=4) 
#            print(min_len)
#            print(num_frags)
#            print(reps)

    # compute cost function for each of the 3**4 combinations
    for num_frags in range(num_frags_range[0], num_frags_range[1] + 1):
        for reps in range(1, int(num_frags / 2)):
                tmp = eu.fragment_timeseries.split_timeseries(series, num_frags)
                untrans, trans, error = eu.initial_conditions.choose_untrans_trans(
                    tmp, reps, report=True)
                ll = cost(error)
                if ll < min_cost:
                    min_cost = ll
                    best_params = (num_frags, reps)

    return best_params, ll
