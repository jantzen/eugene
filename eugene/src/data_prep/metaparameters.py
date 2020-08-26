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
    error_full_series = cost(error)
    # compute leave-one-out errors
    sum_of_loo_errors = 0
    for ii in range(len(series)):
        loo_series = []
        for jj, ss in enumerate(series):
            if not jj == ii:
                loo_series.append(ss)
        tmp = eu.fragment_timeseries.split_timeseries(loo_series, num_frags)
        untrans, trans, error = eu.initial_conditions.choose_untrans_trans(
            tmp, reps, report=True)
        sum_of_loo_errors += cost(error)

    ll = sum_of_loo_errors + error_full_series

    return [num_frags, reps, ll]


def pareto_optimal(option_dictionary):
    tmp = sorted(option_dictionary, key=itemgetter(1))
    ordered = sorted(tmp, key=itemgetter(2))
    min_cost = ordered[0][2]
    for option in ordered:
        if option[2] == min_cost:
            best_params = tuple(option[:2])
    return best_params, min_cost


def tune_ic_selection(
        series, # a list of timeseries to be tested
        num_frags_range=None, # a list giving high and low values for fragments
        min_reps = 5, # the minimum number of reps to select
        max_reps = None, # the maximum number of replicates to consider
        min_len=10, # the minimum tolerable fragment length
        max_len=None, # the maximum tolerable fragment length
        parallel_compute=True, # switch for use of multithreading
        free_cores=2, # cores to leave free when multithreading
        warnings=True # indicates whether or not to display warnings
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
    # deal with warnings
    if not warnings:
        import warnings
        warnings.simplefilter("ignore")

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
        lo = int(min_series_len / max_len) 
        hi = int(min_series_len /  min_len)  
        num_frags_range = [lo, hi]
    if max_reps  is None:
        max_reps = num_frags_range[1] / 10
    print("num_frags_range = {}".format(num_frags_range))
    print("min reps = {}".format(min_reps))
    print("max reps = {}".format(max_reps))

    # enter loop
    min_cost = np.inf
    best_params = ()

    if parallel_compute:
        cpus = max(multiprocessing.cpu_count() - free_cores, 1)
        out = Parallel(n_jobs=cpus,
                verbose=5)(delayed(tune_ic_loop_func)(num_frags, reps, series) 
                    for num_frags in range(num_frags_range[0], num_frags_range[1] + 1)
                    for reps in range(min_reps, max_reps + 1))
        # find a Pareto optimal solution (the most reps at the lowest cost)
        best_params, min_cost = pareto_optimal(out)

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
