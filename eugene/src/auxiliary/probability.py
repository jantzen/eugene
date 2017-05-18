# probability.py

import scipy
import numpy as np
from scipy.integrate import quad

################################################################################
# Functions:
# Phi
# T
# SkewNorm
# SampleSkewNorm
# HellingerDistance
################################################################################

def Phi(x, m, s, a):
    return 0.5 * (1. + scipy.special.erf((x - m) / s / pow(2, 0.5)))


def T(h, a):
    f = lambda x: np.exp(-0.5 * pow(h, 2) * (1 + pow(x, 2))) / (1 + pow(x,2))
    temp = scipy.integrate.quad(f, 0, a)[0]
    return 1. / (2. * np.pi) * temp


def SkewNorm(x, m, s, a):
    return Phi(x, m, s, a) - 2 * T((x - m)/s, a)


def SampleSkewNorm(m, s, a):
    """ A quick and dirty implementation of a skew-normal random variable.
    Returns values from a skew-normal distribution with location m, scale s, and
    shape parameter a (see ). When a = 0, this is just a Gaussian with mean m
    and standard deviation, s.
    """
    # first, choose a random value in [0,1]:
    p = np.random.rand()

    # next, find the value of x corresponding that cumulative probability for
    # the skew-normal
    func = lambda x: p - SkewNorm(x, m, s, a)
    x = scipy.optimize.newton(func, 0)

    return x


def HellingerDistance(dist1, dist2, x_range=[-10,10]):    
    """ Computes the Hellinger distance between two univariate probability
    distributions, dist1 and dist2.
        inputs:
            dist1: a function that returns the probability of x
            dist2: a function that returns the probability of x
            x_range: a list of [low, high] values indicating the integration
            interval with respect to x (for computing the Hellinger distance)
	outputs:
            a scalar value representing the Hellinger distance.
    """

    func = lambda x: (np.sqrt(dist1(x) * dist2(x))).reshape(-1, 1)

    out = quad(func, x_range[0], x_range[1]) 

    h2 = 1. - out[0]

    hellinger = np.sqrt(h2)

    return hellinger

