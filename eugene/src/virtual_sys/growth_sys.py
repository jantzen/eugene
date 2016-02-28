import numpy as np
import scipy

class LogisticGrowthModel(object):
    """Simulates an arbitrary member of the class of logistic equations.
    """

    def __init__(self, r, init_x, K, alpha, beta, gamma, init_t):
        # set the initial population based on passed data
        if init_x > 0:
            self._x = float(init_x)
        else:
            raise ValueError('Invalid initial population assignment. Must be greater than 0')

        # set the time corresponding to that initial concentration
        self._time = init_t
        self._init_t = init_t

        #set remaining attributes
        self._K = float(K)
        self._r = float(r)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)
        self._init_x = float(init_x) 
 
    
    def update_x(self, elapsed_time):
        func = lambda x,t: self._r * x**self._alpha * (1. - (x /
            self._K)**self._beta)**self._gamma

        t = np.array([0., elapsed_time])
        x = scipy.integrate.odeint(func, self._x, t)
        self._x = float(x[1])


    def reset(self):
        self._x = self._init_x
        self._time = self._init_t
