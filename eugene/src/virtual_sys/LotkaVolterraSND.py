# LotkaVolterraSND.py

""" Can simulate a n-species stochastic Lotka-Volterra system.
    System is based on 2-species stochastic model presented in Permanance of 
    Stochastic Lotka-Volterra Systems - Liu, M. & Fan M.; J Nonlinear Sci
    (2017)
"""

import random
import numpy as np
import scipy.integrate


class LotkaVolterraSND( object ):
    """ Implementation of stochastic N species Competitive Lotka-Volterra 
        equations.        
    """

    def __init__(self, r, k, alpha, sigma, init_x, init_t=0):
        """ Initializes a stochastic competitive Lotka-Volterra model with n 
            species
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            k -- an array of species carrying capacities, where k[i] is the 
               carrying capacity of species i. 
            alpha -- the interaction matrix; a matrix of inter-species
                interaction terms, where a[i,j] is the effect of species j on
                the population of species i.
            sigma -- an array of noise intensities where s[i] is the intensity
                of noise affecting species i.
            init_x -- an array of species population size at the start of the
                observation period, where init_x[i] is the initial population
                of species i.
            init_t -- the time index at which the observation period starts.
                (default 0)
        """
        
        # set attributes
        self._r = r
        self._alpha = alpha
        self._k = k

        self._init_x = init_x
        self._init_t = float(init_t)

        self._x = init_x
        self._time = float(init_t)
        self._delta_t = 1
        
        self._sigma = sigma
        

    def update_x(self, elapsed_time):

        t = np.array([0., elapsed_time])
        #t = np.arange(0, elapsed_time) 
        self._delta_t = elapsed_time
        #self._noise = np.random.rand(np.size(self._r), elapsed_time)
        out = scipy.integrate.odeint(self.deriv, self._x, t)
        self._x = out[-1]
        

    def deriv(self, X, t):
        terms = np.zeros(len(X))
        
        for i in range(len(X)):
            noise = np.random.normal()
            #noise = self._noise
            terms[i] = self._r[i] * X[i] * (1 - (np.sum(self._alpha[i] * X)/self._k) ) + (self._sigma[i] * X[i] * noise / (2 * np.sqrt(self._delta_t) ) ) + (self._sigma[i]**2 / 2) * X[i] * (noise**2 - 1)
        return terms
