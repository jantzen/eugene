# resample.py

""" Collection of methods for specialized ramdom sampling.
"""

import numpy as np
from numpy.random import random_sample


def uniform(data, samples=None, bounds=None):
    """ Approximates a random sample from a uniform distribution between
    bounds[0] and bounds[1] by making "samples" number of draws from data.
    Assumes that the variable of interest is in the first column.
    """

    if samples is None:
        samples = 10 * len(data)

    if bounds is None:
        bounds = [np.amin(data[:,0]), np.amax(data[:,0])]

    if bounds[0] > bounds[1]:
        raise ValueError('lower bound {0} is not less than upper bound {1}'.format(
            bounds[0],bounds[1]))

    # draw random samples from the specified half-open interval
    rs = (bounds[1] - bounds[0]) * random_sample(samples) + bounds[0]

    resampled_data = []

    sorted_ind = np.argsort(data[:,0])
    data_sorted = data[sorted_ind]
    # get approximate indeces of closest values in data
    inds = np.searchsorted(data_sorted[:,0], rs)
    for i, r in zip(inds, rs):
        # get the index of the closest value in data
        if i == 0:
            resampled_data.append(data_sorted[i,:])
        elif i == len(data_sorted):
            resampled_data.append(data_sorted[-1,:])
        else:
            before = data_sorted[i-1, 0]
            after = data_sorted[i, 0]
            if after - r < r - before:
                resampled_data.append(data_sorted[i, :])
            else:
                resampled_data.append(data_sorted[i-1, :])

    return np.array(resampled_data)