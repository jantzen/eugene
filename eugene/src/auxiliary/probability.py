# probability.py

import scipy
import numpy as np
from scipy.integrate import quad, dblquad
import pdb

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


def HellingerDistance(dist1, dist2, x_range=[-np.inf,np.inf]):    
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


def Hellinger2D(dist1, dist2, x_low=-np.inf, x_high=np.inf, y_low=None,
        y_high=None):    
    """ Computes the Hellinger distance between two bivariate probability
    distributions, dist1 and dist2.
        inputs:
            dist1: a function that returns the probability of x, y
            dist2: a function that returns the probability of x, y
            x_low: float indicating the lower bound of the integration
            interval with respect to x (for computing the Hellinger distance)
	    x_high: float indicating the upper bound of the integration
            interval with respect to x (for computing the Hellinger distance)
	    y_low: a callable function describing the lower boundary curve of
            y as a function of x
	    y_high: a callable function describing the upper boundary curve of
            y as a function of x
        outputs:
            a scalar value representing the Hellinger distance.
    """

    if y_low == None:
        y_low = lambda x: x_low

    if y_high == None:
        y_high = lambda x: x_high

    if type(y_low) == np.float64 or type(y_low)==float:
        y_low = lambda x, y_low=y_low: y_low

    if type(y_high) == np.float64 or type(y_high)==float:
        y_high = lambda x, y_high=y_high: y_high
    
    func = lambda x,y: (np.sqrt(dist1(x,y) * dist2(x,y))).reshape(-1, 1)

    out = dblquad(func, x_low, x_high, y_low, y_high) 

    hellinger = 1. - np.sqrt(out[0])

    return hellinger


def EuclideanDistance(dist1, dist2, x_range=[-10,10]):
    """ Computes the Euclidean (L2) distance between two univariate probability
    distributions, dist1 and dist2.
        inputs:
            dist1: a function that returns the probability of x
            dist2: a function that returns the probability of x
            x_range: a list of [low, high] values indicating the integration
            interval with respect to x (for computing the Hellinger distance)
	outputs:
            a scalar value representing the L2 distance.
    """
    
    func = lambda x: (dist1(x) * dist2(x)).reshape(-1, 1)
    
    out = quad(func, x_range[0], x_range[1])
    
    l2 = np.sqrt(out)
    
    return l2
