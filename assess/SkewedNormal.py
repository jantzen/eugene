# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from scipy import special
from scipy.special import *


class gaussian_gen(stats.rv_continuous):
    "normal/Gaussian distribution"
    def __init__(self, x):
        self.x = x
    
    def _pdf(self, x):
        return exp(-x**2 / 2.) / sqrt(2.0 * pi)
        

# We need something like the thing below for the DeviantNoiseExperiment, under 
# construction obviously
class skewed_normal(stats.rv_continuous):
    # Towards the sampler doodad, for grabbing a value from a SND    
    # Builds a skewed normal distribution
    def __init__(self, skew):
        self.skew = skew
        
    def pdf(x):
        # standard normal density function
        numerator = exp(-x**2/2)
        denominator = 1/sqrt(2*pi) 
        return numerator / denominator 
    
    def cdf(x):
        # distribution function for the standard normal
        return (1 + erf(x/sqrt(2))) / 2
        
    def skewed(self, x, skew):
        # set up the pdf for the skewed normal distribution.
        # f(x) = 2 * pdf(x) * cdf(skew * x)   [skew is a shape parameter]
        
        return 2 * skewed_normal.pdf(x) * skewed_normal.cdf(x * skew)

billyjoe = skewed_normal(2)