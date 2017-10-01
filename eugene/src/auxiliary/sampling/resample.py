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

    if samples == None:
        samples = 10 * len(data)

    if bounds == None:
        bounds = [np.amin(data[:,0]), np.amax(data[:,0])]

    if bounds[0] > bounds[1]:
        raise ValueError('lower bound must be less than upper bound')

    # draw random samples from the specified half-open interval
    rs = (bounds[1] - bounds[0]) * random_sample(samples) + bounds[0]

    resampled_data = []

    for x in rs:
        # get the index of the closest value in data
        ind = np.argmin(np.abs(data[:,0] - x))
        resampled_data.append(data[ind,:])

    return np.array(resampled_data).reshape(samples,2)
