# probability.py

import scipy
import numpy as np

################################################################################
# Functions:
# Phi
# T
# SkewNorm
# SampleSkewNorm
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
