# LotkaVolterraND.py

import numpy as np
import scipy.integrate


class LotkaVolterraNDLin(object):
    """ Implementation of N species Competitive Lotka-Volterra equations.

    """


    def __init__(self, r, k, alpha, init_x, d, init_t=0):
        """ Initializes a competitive Lotka-Volterra model with n species

            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            k -- an array of species carrying capacities, where k[i] is the
               carrying capacity of species i.
            a -- the interaction matrix; a matrix of inter-species interaction
                terms, where a[i,j] is the effect of species j on the
                population of species i.
            init_x -- an array of species population size at the start of the
                observation period, where init_x[i] is the initial population
                of species i.
            d -- a scalar that modifies the impact of the x**2 term, and alpha.
            init_t -- the time index at which the observation period starts.
                (default 0)
        """

        # set attributes
        self._r = r
        self._alpha = alpha
        self._k = k
        self._d = d

        self._init_x = init_x
        self._init_t = float(init_t)

        self._x = init_x
        self._time = float(init_t)


    def update_x(self, elapsed_time):
        # deriv = lambda X, t: np.array([self._r1 * X[0] * (1 - (X[0] +
        #    self._alpha1 * X[1]) / self._k1), self._r2 * X[1] * (1 - (X[1] +
        #    self._alpha2 * X[0]) / self._k2)])

        t = np.array([0., elapsed_time])
        out = scipy.integrate.odeint(self.deriv, self._x, t)
        self._x = out[1]


    def deriv(self, X, t):
        terms = np.zeros(len(X))

        for i in range(len(X)):
            terms[i] = self._r[i] * X[i] * (
                        1 - (np.sum(self._alpha[i] * self._d * X) / self._k[i]))

        return terms


    def check_xs(self, times):
        t_n = 0.
        xs = np.array(self._x).reshape(1, len(self._x))
        for i in range(len(times)):
            if times[i] == 0.:
                continue
            interval = times[i] - t_n
            t_n = times[i]
            self.update_x(interval)
            xs = np.vstack((xs, self._x.reshape(1, len(self._x))))

        return xs

