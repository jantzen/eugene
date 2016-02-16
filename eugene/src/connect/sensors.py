import math
import time
import copy
import random
import numpy as np

#Parent Sensor
class VABSensor( object ):
    def __init__(self, dynamic_range):
        self._init_value = None
        self._range = dynamic_range

    def get_range(self):
        return self._range

###############################################
###############################################
#Particular Sensors

#do we want Virtual and Real separate OR
# Virtual next to Real?

class VABTimeSensor(VABSensor):
    def read(self, sys):
        return sys.update_time()


class VABConcentrationSensor(VABSensor):
    def __init__(self, dynamic_range, noise_stdev=0, proportional=False):
        self._range = dynamic_range
        self._noise_stdev = noise_stdev
        self._proportional = proportional
    
    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
#        else:
#            if self._noise_stdev == 0:
#                concentration = sys.update_x()
#            elif self._proportional:
#                x = sys.update_x()
#                noise = np.random.normal(0, self._noise_stdev * x)
#                concentration = x + noise
#            else:
#                concentration = sys.update_x() + np.random.normal(0,
#                        self._noise_stdev)
#            if concentration > self._range[1] or concentration < self._range[0]:
#                return 'outofrange'
#            else:
#                return concentration
        else:
            if self._noise_stdev == 0:
                concentration = sys._x
            elif self._proportional:
                x = sys._x
                noise = np.random.normal(0, self._noise_stdev * x)
                concentration = x + noise
            else:
                concentration = sys._x + np.random.normal(0,
                        self._noise_stdev)
            if concentration > self._range[1] or concentration < self._range[0]:
                return 'outofrange'
            else:
                return concentration


class VABVoltageSensor(VABSensor):
    def __init__(self, dynamic_range, noise_stdev=0, proportional=False):
        self._range = dynamic_range
        self._noise_stdev = noise_stdev


    def read(self, sys):
        
        if len(self._range) != 9:       # Is the right range (0-10) specified? 
            raise ValueError('No sensor range specified.')
        else:
            # start with what our noise looks like.     
            if self._noise_stdev == 0:
                voltage = sys._v
            elif self.proportional:  # if the noise is proportional...
                v = sys._v
                noise = np.random.normal(0, self._noise_stdev * v)
                voltage = v + noise
            else: 
                voltage = sys._v + np.random.normal(0,self._noise_stdev)   
            # now check that the voltage falls within sensor range
            if voltage > self._range(9) or voltage < self._range(0):
                return 'Error, that\'s not in range!'
            else:
                return voltage
                

class PopulationSensor(VABSensor):
    def __init__(self, dynamic_range, noise_stdev=0, proportional=False):
        self._range = dynamic_range
        self._noise_stdev = noise_stdev
        self._proportional = proportional
    
    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            if self._noise_stdev == 0:
                population = sys._x
            elif self._proportional:
                x = sys._x
                noise = np.random.normal(0, self._noise_stdev * x)
                population = x + noise
            else:
                population = sys._x + np.random.normal(0,
                        self._noise_stdev)
            if population > self._range[1] or population < self._range[0]:
                return 'outofrange'
            else:
                return population


class CCVoltageSensor(VABSensor):
    """ For use with simulated chaotic circuits. The passed value deriv
    indicates which derivative of voltage the sensor should measure.
    """
    def __init__(self, dynamic_range, deriv, noise_stdev=0, proportional=False):
        self._range = dynamic_range
        self._noise_stdev = noise_stdev
        self._proportional = proportional
        self._deriv = deriv
 
    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            if self._noise_stdev == 0:
                out = sys._x[self._deriv]
            elif self._proportional:
                x = sys._x[self._deriv]
                noise = np.random.normal(0, self._noise_stdev * x)
                out = x + noise
            else:
                out = sys._x[self._deriv] + np.random.normal(0,
                        self._noise_stdev)
            if out > self._range[1] or out < self._range[0]:
                return 'outofrange'
            else:
                return out

       
