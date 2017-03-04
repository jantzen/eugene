# chaotic_circuits.py

""" Simulates an electrical circuit with chaotic features, pulling the 
    information for a given circuit from circuitparams. The jerk equation is 
    expressed as a system of ODE's that is passed to scipy's odeint integrator.
"""
 

from . import circuitparams
import numpy as np
import scipy.integrate


class ChaoticCircuit( object ):
    """ Simulates a chaotic ccircuit with a varying voltage described by a "jerk
    equation."
    """

    def __init__(self, circNum, init_x=None, init_t=0):
        # pull the dictionary of circuit parameters
        cdict = circuitparams.cdict

        # set attributes
        if type(circNum) == int and circNum < len(cdict):
            self._Ais = cdict[circNum][0]
        else:             
            print("unknown equation id")

        if init_x == None:
            self._init_x = cdict[circNum][1]
        else:
            self._init_x = float(init_x)

        self._x = self._init_x
        self._init_t = init_t
        self._time = self._init_t

    def update_x(self, elapsed_time):
        Ai = self._Ais
        phi = Ai[7]
        deriv = lambda x,t: np.array([x[1], x[2], Ai[0] * x[2] + Ai[1] * phi(x[2]) \
            + Ai[2] * x[1] + Ai[3] * phi(x[1]) + Ai[4] * x[0] \
            + Ai[5] * phi(x[0]) + Ai[6]])
        t = np.array([0., elapsed_time])
        x = scipy.integrate.odeint(deriv, self._x, t) 
        self._x = x[1]


    def reset(self):
        self._x = self._init_x
        self._time = self._init_t
