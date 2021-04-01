# StochasticTools.py

""" Methods for randomly sampling under contraints, etc.
"""

import numpy as np

def drawMean(num_vars, var_range, mean, tol=0):
    """ Randomly samplet a set of num_vars numbers with the desired mean.
    """

    sample = []
    a = var_range[0]
    b = var_range[1]
    
    s = 0
    for i in range(num_vars):
        min_xi = max([num_vars * mean - s - b * (num_vars - 1 - i), a])
        max_xi = min([num_vars * mean - s - a * (num_vars - 1 - i), b])
        xi = np.random.uniform(min_xi, max_xi)
        sample.append(xi)
        s = s + xi

    return sample
