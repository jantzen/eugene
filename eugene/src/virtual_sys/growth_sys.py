import numpy as np
import scipy

class LogisticGrowthModel(object):
    """Simulates an arbitrary member of the class of logistic equations.
    """

    def __init__(self, r, init_x, K, alpha, beta, gamma, init_t,
            stochastic=False, sigma=0.1, steps=1000):

        # set a flag indicating whether the dynamics is stochastic or
        # deterministic
        self._stochastic = stochastic
        self._sigma = sigma
        self._steps = steps

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
        if self._stochastic:
            if elapsed_time == 0.:
                return None

            delta = float(elapsed_time) / float(self._steps)
            X = self._x
            for s in range(self._steps):
                noise = np.random.normal()
                
                dX = self._r * X**(self._alpha) * (1 - (X/self._K)**(self._beta)
                        )**self._gamma + (self._sigma * X * noise / (2. *
                np.sqrt(delta) ) ) + (self._sigma**2 / 2.) * (X
                * (noise**2 - 1.))

                X = X + dX * delta
 
                X = np.max([X, 0.])

            self._x = X
#            self._x = float(x[1]) + (np.random.normal(0., 0.5)**2 * elapsed_time /
#                self._r / (1 + elapsed_time) * self._K)
        else:
            func = lambda x,t: self._r * x**self._alpha * (1. - (x /
                self._K)**self._beta)**self._gamma

            t = np.array([0., elapsed_time])
            x = scipy.integrate.odeint(func, self._x, t)


            self._x = float(x[1])


    def reset(self):
        self._x = self._init_x
        self._time = self._init_t


class GompertzGrowthModel(object):
    """Simulates an arbitrary member of the class of Gompertz models.
    """

    def __init__(self, a, b, init_x, init_t, stochastic=False):

        # set a flag indicating whether the dynamics is stochastic or
        # deterministic
        self._stochastic = stochastic

        # set the initial population based on passed data
        if init_x > 0:
            self._x = float(init_x)
        else:
            raise ValueError('Invalid initial population assignment. Must be greater than 0')

        # set the time corresponding to that initial concentration
        self._time = init_t
        self._init_t = init_t

        #set remaining attributes
        self._a = float(a)
        self._b = float(b)
        self._init_x = float(init_x) 
 
    
    def update_x(self, elapsed_time):
        func = lambda x,t: self._b * x * np.exp(self._a - self._b * t)

        t = np.array([self._time, self._time + elapsed_time])
        x = scipy.integrate.odeint(func, self._x, t)

        if self._stochastic:
            pass
        else:
            self._x = float(x[1])
            self._time += elapsed_time


    def reset(self):
        self._x = self._init_x
        self._time = self._init_t
