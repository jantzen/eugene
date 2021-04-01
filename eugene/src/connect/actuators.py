import math
import random
import numpy as np


class VABSensor( object ):
    def __init__(self, dynamic_range):
        self._init_value = None
        self._range = dynamic_range

    def get_range(self):
        return self._range


#Virtual Actuators
class VABVirtualTimeActuator(VABSensor):
    def __init__(self, initial_time=0):
        self._initial_time = initial_time
        self._time = initial_time

    def set(self, sys, delta):
        sys._time = sys._time + delta
        sys.update_x(delta)

    def time(self, sys):
        return sys._time
    

class VABConcentrationActuator(VABSensor):
    def set(self, sys, value):
        sys._x = value
        sys._time = sys._init_t


class PopulationActuator(VABSensor):
    def set(self, sys, value):
        sys._x = value
        sys._time = sys._init_t


class CCVoltageActuator(VABSensor):
    """ For use with simulated chaotic circuits. The passed parameter 'deriv'
        indicates which derivative of the voltage is to be set.
    """
    def __init__(self, dynamic_range, deriv):
        self._init_value = None
        self._range = dynamic_range   
        self._deriv = deriv

    def set(self, sys, value):
        sys._x[self._deriv] = value
        sys._time = sys._init_t    


class LorenzActuator(VABSensor):
    def __init__(self, variable, dynamic_range):
        if not variable in set(['x','y','z']):
            raise ValueError('Inorrect variable specification. Must be string "x", "y", or "z".')
        else:
            self._variable = variable
        self._range = dynamic_range
 
    def set(self, sys, value):
        exec('sys._' + self._variable + ' = value')
        sys._time = sys._init_t


class LotkaVolterra2DActuator(VABSensor):
    def __init__(self, variable, dynamic_range):
        if not variable in set([1,2]):
            raise ValueError('Inorrect variable specification. Must be 1 or 2.')
        else:
            self._variable = variable
        self._range = dynamic_range
 
    def set(self, sys, value):
        exec('sys._x' + str(self._variable) + ' = value')
        sys._time = sys._init_t


class LotkaVolterraNDActuator(VABSensor):
    def __init__(self, variable, dynamic_range):
        self._variable = variable
        self._range = dynamic_range
 
    def set(self, sys, value):
        sys._x[self._variable-1] = value
#        exec('sys._x' + str(self._variable) + ' = value')
        sys._time = sys._init_t
