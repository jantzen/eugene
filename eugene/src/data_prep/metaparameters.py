# file: metaparameters.py

import eugene as eu
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from operator import itemgetter
import copy
import pdb


def cost(error_matrix):
    m = np.copy(error_matrix)
    m[np.where(m==2)] = 1
    m[np.where(m==3)] = 2
    return np.sum(m)


def tune_ic_loop_func(num_frags, reps, series, alpha, beta, mu_spec):
    tmp = eu.fragment_timeseries.split_timeseries(series, num_frags)
    untrans, trans, error = eu.initial_conditions.choose_untrans_trans(
        tmp, reps, alpha=alpha, beta=beta, mu_spec=mu_spec, report=True)
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
            tmp, reps, alpha=alpha, beta=beta, mu_spec=mu_spec, report=True)
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
        alpha=0.5,
        beta=0.2,
        mu_spec=None,
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
        max_len = int(2 * min_len)
    if num_frags_range is None:
        lo = int(min_series_len / max_len) 
        hi = int(min_series_len /  min_len)  
        num_frags_range = [lo, hi]
    if max_reps  is None:
        max_reps = int(num_frags_range[1] / 10)
    print("num_frags_range = {}".format(num_frags_range))
    print("min reps = {}".format(min_reps))
    print("max reps = {}".format(max_reps))

    # enter loop
    min_cost = np.inf
    best_params = ()

    if parallel_compute:
        cpus = max(multiprocessing.cpu_count() - free_cores, 1)
        out = Parallel(n_jobs=cpus,
                verbose=5)(delayed(tune_ic_loop_func)(num_frags, reps, series, alpha, beta, mu_spec) 
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
                        tmp, reps, alpha=alpha, beta=beta, mu_spec=mu_spec, report=True)
                    ll = cost(error)
                    if ll < min_cost:
                        min_cost = ll
                        best_params = (num_frags, reps)
                    if min_cost == 0:
                        break
            if min_cost == 0:
                break

    return best_params, min_cost

def tune_offsets(
        series,     # a dictionary of timeseries as d x p np-arrays
        num_frags,      # approximate number of fragments into which to divide each timeseries
        reps,       # the number of replicates to select for each timeseries
        alpha=0.5,
        beta=0.2,
        mu_spec=None,
        warnings=True # indicates whether or not to display warnings
        ):

    # deal with warnings
    if not warnings:
        import warnings
        warnings.simplefilter("ignore")

    # determine the best fragment length 
    lengths = []
    for key in sorted(series.keys()):
        ll = series[key].shape[1]
        lengths.append(ll)
    min_len = min(lengths)
    frag_length = int(np.ceil(min_len / num_frags))

    # construct the set of fragments for each time series
    series_list = []
    for key in sorted(series.keys()):
        series_list.append(series[key])
    frags = eu.fragment_timeseries.fixed_length_frags(series_list, frag_length)

    # construct the set of initials for each time series
    series_initials = dict([])
    for ii, key in enumerate(sorted(series.keys())):
        initials = []
        tmp = frags[ii]
        for seg in tmp:
            initials.append(seg[:,0].reshape(-1, 1))
        series_initials[key] = np.concatenate(initials, axis=1)
        
    # identify the two sets of initials whose means are the farthest apart
    max_distance = -1.
    for ii, key1 in enumerate(sorted(series_initials.keys())):
        for jj, key2 in enumerate(sorted(series_initials.keys())):
            if not jj == ii:
                mu1 = np.mean(series_initials[key1], axis=1)
                mu2 = np.mean(series_initials[key2], axis=1)
                distance = np.linalg.norm(mu1 - mu2)
                if distance > max_distance:
                    max_distance = distance
                    gd = [ii, jj]
                    gd_keys = [key1, key2]
    
    # loop to fix offset for the first pair of sets
#    frag_length = frags[gd[0]][0].shape[1]
    print("Frag length: {}".format(frag_length))
    best_offset = 0
    min_cost = np.inf
    for offset in range(frag_length):
        data2 = series[gd_keys[1]][:,offset:]
        data1 = series[gd_keys[0]][:,:data2.shape[1]]
        tmp = eu.fragment_timeseries.fixed_length_frags([data1, data2],
                frag_length)
        tmp = eu.fragment_timeseries.trim(tmp)
        untrans, trans, error = eu.initial_conditions.choose_untrans_trans(
            tmp, reps, alpha=alpha, beta=beta, mu_spec=mu_spec, report=True)
        ll = cost(error)
        if ll < min_cost:
            min_cost = ll
            best_offset = offset
    offsets = dict([])
    costs = dict([])
    offsets[gd_keys[0]] = 0
    offsets[gd_keys[1]] = best_offset
    costs[gd_keys[0]] = 0
    costs[gd_keys[1]] = min_cost

    remaining_keys = sorted(series.keys())
    remaining_keys.remove(gd_keys[0])
    remaining_keys.remove(gd_keys[1])

    # loop over the remaining sets and possible offsets for each to find the
    # best overall combination (not necessarily the global best)
    data = []
    data.append(series[gd_keys[0]][:,offsets[gd_keys[0]]:])
    data.append(series[gd_keys[1]][:,offsets[gd_keys[1]]:])
    for key in remaining_keys:
        best_offset = 0
        min_cost = np.inf
        for offset in range(frag_length):
            tmp_data = copy.deepcopy(data)
            tmp_data.append(series[key][:,offset:])
            # trim all of the data to the same length
            min_len = np.inf
            for td in tmp_data:
                series_len = td.shape[1]
                if series_len < min_len:
                    min_len = series_len
            for ii, td in enumerate(tmp_data):
                tmp_data[ii] = td[:, :min_len]
            tmp = eu.fragment_timeseries.fixed_length_frags(tmp_data,
                    frag_length)
            tmp = eu.fragment_timeseries.trim(tmp)
            untrans, trans, error = eu.initial_conditions.choose_untrans_trans(
                tmp, reps, alpha=alpha, beta=beta, mu_spec=mu_spec, report=True)
            ll = cost(error)
            if ll < min_cost:
                min_cost = ll
                best_offset = offset
        offsets[key] = best_offset
        costs[key] = min_cost
        data.append(series[key][:,offsets[key]:])
  
    return offsets, costs, frag_length


def apply_offsets(offsets, data):
    offset_data = dict([])
    for key in sorted(offsets.keys()):
        offset = offsets[key]
        offset_data[key] = data[key][:, offset:]
 
    # clip all time series to the same length
    kk = sorted(offset_data.keys())
    min_series_len = offset_data[kk[0]].shape[1]
    for kk in sorted(offset_data.keys()):
        ll = offset_data[kk].shape[1]
        if ll < min_series_len:
            min_series_len = ll
 
    print('Clipping all data to length {}...'.format(min_series_len))
    for key in sorted(offset_data.keys()):
        data = offset_data[key][:, :min_series_len]
        offset_data[key] = data
 
    return offset_data
