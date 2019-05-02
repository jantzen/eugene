# LotkaVolterraND.py

import numpy as np
import scipy.integrate


class LotkaVolterra2OND( object ):
    """ Implementation of N species Competitive Lotka-Volterra equations.
        
    """

    def __init__(self, r, k, alpha, init_x, init_y, ord_scale, init_t=0):
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
            init_y -- an array of growth velocities at the start of the
                observation period, where init_y[i] is the growth velocity of
                species i.
            ord_scale -- a scalar that determines how much a species' growth
                velocity impacts its overall growth.
            init_t -- the time index at which the observation period starts.
                (default 0)
        """
        
        # set attributes
        self._r = r
        self._alpha = alpha
        self._k = k
        self._ord_scale = ord_scale

        self._init_x = init_x
        self._init_y = init_y
        self._init_t = float(init_t)

        self._y = init_y
        self._x = init_x
        self._time = float(init_t)
        

    def update_x(self, elapsed_time):
        t = np.array([0., elapsed_time])
        y = np.array([self._x, self._y]).flatten()
        y = np.squeeze(y)
#        print(y)
        out = scipy.integrate.odeint(self.deriv, y, t)
        ys = np.array_split(out[1], 2)
        self._x = ys[0]
        self._y = ys[1]
        

    def deriv(self, y, t):
        # let X be the vector [pop[i], y[i]]
        ys = np.array_split(y, 2)
        X = ys[0]
        Y = ys[1]
        terms = np.zeros((2, len(X)))
        
        for i in range(len(X)):
            # xi' = y
            # yi' = ri*xi * (1 - ri*xi*SUM(ani*xn)/ki - y)
            # Above, not below.
            # xi' = ri*xi * (1 - ri*xi*SUM(ani*xn)/ki)
            # xi'' = (rn - ani*xi)(ri*xi - xi') + (1/xi)(xi')^2

            terms1 = Y[i]
            terms2 = self._ord_scale*(self._r[i] * X[i] * (1 - (np.sum(self._alpha[i]*X)/self._k[i])) - Y[i])
            terms[0, i] = terms1
            terms[1, i] = terms2
            
        terms = terms.flatten()
#        print(terms)
        return terms
        
    
    def check_xs(self, times):
        t_n = 0.
        xs = np.array(self._x).reshape(1, len(self._x))
        ys = np.array(self._y).reshape(1, len(self._y))
        out = np.array([xs, ys]).flatten()
        for i in range(len(times)):
            if times[i] == 0.:
                continue
            interval = times[i] - t_n
            t_n = times[i]
            self.update_x(interval)
            xt = self._x.reshape(1, len(self._x))
            yt = self._y.reshape(1, len(self._y))
            outt = np.array([xt, yt]).flatten()
            out = np.vstack((out, outt))
            xs = np.vstack((xs, self._x.reshape(1, len(self._x))))
            ys = np.vstack((ys, self._y.reshape(1, len(self._y))))
        
        return out
        
