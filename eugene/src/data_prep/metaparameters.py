# file: metaparameters.py

import eugene as eu
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from operator import itemgetter


def cost(error_matrix):
    m = np.copy(error_matrix)
    m[np.where(m==2)] = 1
    m[np.where(m==3)] = 2
    return np.sum(m)


def tune_ic_loop_func(num_frags, reps, series):
    tmp = eu.fragment_timeseries.split_timeseries(series, num_frags)
    untrans, trans, error = eu.initial_conditions.choose_untrans_trans(
        tmp, reps, report=True)
    ll = cost(error)

    return [num_frags, reps, ll]


def tune_ic_selection(
        series, # a list of timeseries to be tested
        num_frags_range=None, # a list giving high and low values for fragments
        min_reps = 5, # the minimum number of reps to select
        min_len=10, # the minimum tolerable fragment length
        max_len=None, # the maximum tolerable fragment length
        parallel_compute=True, # switch for use of multithreading
        free_cores=2 # cores to leave free when multithreading
        ):

    """ 
        Purpose:
        Sets optimal values for the following metaparameters:
        num_frags: number of fragments into which to split each timeseries
        reps: number of replicates to select from each pool

        Method:
        Performs a brute-force combinatorial search over discrete parameter
        space as bounded by the given ranges in order to minimize the total
        number of errors (i.e., statistically significant differences in the
        distribution of initial values for the selected untrans and trans sets
        drawn from the fragmented time series.
    """

    # gather info about input
    min_series_len = series[0].shape[1] 
    for ts in series:
        tmp = ts.shape[1]
        if  tmp < min_series_len:
            min_series_len = tmp
    print("min_series_len = {}".format(min_series_len))

    # set up ranges
    if max_len is None:
        max_len = 2 * min_len
    if num_frags_range is None:
        lo = int(min_series_len / (max_len)) 
        hi = int(min_series_len /  min_len)  
        num_frags_range = [lo, hi]
        print("num_frags_range = {}".format(num_frags_range))
        print("max reps = {}".format(int(hi / 2)))

    # enter loop
    min_cost = np.inf
    best_params = ()

    if parallel_compute:
        cpus = max(multiprocessing.cpu_count() - free_cores, 1)
        out = Parallel(n_jobs=cpus,
                verbose=5)(delayed(tune_ic_loop_func)(num_frags, reps, series) 
                    for num_frags in range(num_frags_range[0], num_frags_range[1] + 1)
                    for reps in range(min_reps, int(num_frags / 2)))
        # sort the output to find the lowest cost
        opt = sorted(out, key=itemgetter(2))[0]
        best_params = tuple(opt[:2])
        min_cost = opt[2]

    else:
        # compute cost function for each of the 3**4 combinations
        for num_frags in range(num_frags_range[0], num_frags_range[1] + 1):
            for reps in range(min_reps, int(num_frags / 2)):
                    tmp = eu.fragment_timeseries.split_timeseries(series, num_frags)
                    untrans, trans, error = eu.initial_conditions.choose_untrans_trans(
                        tmp, reps, report=True)
                    ll = cost(error)
                    if ll < min_cost:
                        min_cost = ll
                        best_params = (num_frags, reps)
                    if min_cost == 0:
                        break
            if min_cost == 0:
                break

    return best_params, min_cost
