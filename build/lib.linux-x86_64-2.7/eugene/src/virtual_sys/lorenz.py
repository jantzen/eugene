# lorenz.py

""" Simulates a Lorenz system (in terms of Lorenz's X, Y, and Z variables).
"""

import numpy as np
import scipy.integrate


class LorenzSystem( object ):
    """ Implementation of Lorenz system.
    """

    def __init__(self, beta, sigma, rho, init_x, init_y, init_z, init_t=0):
        # set attributes
        self._beta = beta
        self._sigma = sigma
        self._rho = rho

        self._init_x = float(init_x)
        self._init_y = float(init_y)
        self._init_z = float(init_z)
        self._init_t = float(init_t)

        self._x = float(init_x)
        self._y = float(init_y)
        self._z = float(init_z)
        self._time = float(init_t)

    def update_x(self, elapsed_time):
        deriv = lambda X, t: np.array([self._sigma * (X[1] - X[0]), X[0] *
            (self._rho - X[2]) - X[1], X[0] * X[1] - self._beta * X[2]])

        t = np.array([0., elapsed_time])
        out = scipy.integrate.odeint(deriv, np.array([self._x, self._y, self._z]), t)
        self._x = out[1][0]
        self._y = out[1][1]
        self._z = out[1][2]
