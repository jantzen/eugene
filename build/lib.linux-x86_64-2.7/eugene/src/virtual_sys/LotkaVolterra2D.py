# LotkaVolterra2D.py

""" Simulates a Lorenz system (in terms of Lorenz's X, Y, and Z variables).
"""

import numpy as np
import scipy.integrate


class LotkaVolterra2D( object ):
    """ Implementation of Lorenz system.
    """

    def __init__(self, r1, r2, k1, k2, alpha1, alpha2, init_x1, init_x2, init_t=0):
        # set attributes
        self._r1 = r1
        self._r2 = r2
        self._k1 = k1
        self._k2 = k2
        self._alpha1 = alpha1
        self._alpha2 = alpha2

        self._init_x1 = float(init_x1)
        self._init_x2 = float(init_x2)
        self._init_t = float(init_t)

        self._x1 = float(init_x1)
        self._x2 = float(init_x2)
        self._time = float(init_t)

    def update_x(self, elapsed_time):
        deriv = lambda X, t: np.array([self._r1 * X[0] * (1 - (X[0] +
            self._alpha1 * X[1]) / self._k1), self._r2 * X[1] * (1 - (X[1] +
            self._alpha2 * X[0]) / self._k2)])

        t = np.array([0., elapsed_time])
        out = scipy.integrate.odeint(deriv, np.array([self._x1, self._x2]), t)
        self._x1 = out[1][0]
        self._x2 = out[1][1]
