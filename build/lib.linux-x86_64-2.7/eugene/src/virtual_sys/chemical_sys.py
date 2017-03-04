import math
import time
import copy
import random
import numpy as np


class VABSystemFirstOrderReaction(object):
    """This class defines a simulated first-order chemical reaction: aA ->
    products. Throughout, x is used for concentration, and k for the reaction
    constant (that includes the stoichiometric coefficient, a). """
    
    
    def __init__(self, init_x, k, init_t=0):
        # set the initial population based on passed data
        if init_x > 0:
            self._x = float(init_x)
        else:
            raise ValueError('Invalid initial concentration assignment. Must be greater than 0')

        # set the time corresponding to that initial concentration
        self._time = init_t
        self._init_t = init_t

        #set remaining attributes
        self._k = float(k)
        self._init_x = float(init_x) 
    
    def update_x(self, elapsed_time):
        # now compute the current population based on the model
        x = self._x * math.exp(-self._k * elapsed_time) 
        # update the class variables before returning the values
        self._x = x

    def reset(self):
        self._x = self._init_x
        self._time = self._init_t


class VABSystemSecondOrderReaction(object):
    """This class defines a simulated second-order chemical reaction: aA ->
    products. Throughout, x is used for concentration, and k for the reaction
    constant (that includes the stoichiometric coefficient, a). """
    
    
    def __init__(self, init_x, k, init_t=0):
        # set the initial population based on passed data
        if init_x > 0:
            self._x = float(init_x)
        else:
            raise ValueError('Invalid initial concentration assignment. Must be greater than 0')

        # set the time corresponding to that initial concentration
        self._time = init_t
        self._init_t = init_t

        #set remaining attributes
        self._k = float(k)
        self._init_x = float(init_x) 
        
    def update_x(self, elapsed_time):
        # now compute the current population based on the model
        x = self._x / (1. + self._k * elapsed_time * self._x)
        # update the class variables before returning the values
        self._x = x

    def reset(self):
        self._x = self._init_x
        self._time = self._init_t
 

class VABSystemThirdOrderReaction(object):
    """This class defines a simulated third-order chemical reaction: 
    aA -> products. Throughout, x is used for concentration, and k for the reaction
    constant (that includes the stoichiometric coefficient, a). """
    
    def __init__(self, init_x, k, init_t=0):
        # set the initial population based on passed data
        if init_x > 0:
            self._x = float(init_x)
        else:
            raise ValueError('Invalid initial concentration assignment. Must be greater than 0')

        # set the time corresponding to that initial concentration
        self._time = init_t
        self._init_t = init_t


        #set remaining attributes
        self._k = float(k)
        self._init_x = float(init_x) 
       
    def update_x(self, elapsed_time):
        # now compute the current population based on the model
        x = self._x / math.pow((1. + 2. * self._k * elapsed_time *
            math.pow(self._x,2)),0.5)
        # update the class variables before returning the values
        self._x = x

    def reset(self):
        self._x = self._init_x
        self._time = self._init_t
