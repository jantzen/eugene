import math
import time
import copy
import random
import numpy as np
import eugene as eu
import pdb

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
    def __init__(self, dynamic_range, noise_stdev=0, proportional=False,
                 skew=0):
        self._range = dynamic_range
        self._noise_stdev = noise_stdev
        self._proportional = proportional
        self._skew = skew
    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            if self._noise_stdev == 0:
                population = sys._x
            elif self._proportional:
                x = sys._x
                if self._skew > 0:
                    s = self._noise_stdev * x / np.sqrt(1 - (2. * self._skew**2) /
                            (np.pi * (1. - self._skew**2)))
                    noise = eu.probability.SampleSkewNorm(0, s, self._skew)
                    population = x + noise
                else:
                    x = sys._x
                    noise = np.random.normal(0, self._noise_stdev * x)
                    population = x + noise
            else:
                x = sys._x
                if self._skew > 0:
                    s = self._noise_stdev / np.sqrt(1 - (2. * self._skew**2) /
                            (np.pi * (1. - self._skew**2)))
                    noise = eu.probability.SampleSkewNorm(0, s, self._skew)
                    population = x + noise
                else:                    
                    population = sys._x + np.random.normal(0, self._noise_stdev)
                                                          
                                                           
            if population > self._range[1] or population < self._range[0]:
                return 'out of range'
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


class LorenzSensor(VABSensor):
    def __init__(self, variable, dynamic_range, noise_stdev=0, proportional=False,
                 skew=0):
        if not variable in set(['x','y','z']):
            raise ValueError('Inorrect variable specification. Must be string "x", "y", or "z".')
        else:
            self._variable = variable
        self._range = dynamic_range
        self._noise_stdev = noise_stdev
        self._proportional = proportional
        self._skew = skew

    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            if self._noise_stdev == 0:
                exec('val = sys._' + self._variable)

            elif self._proportional:
                exec('temp = sys._' + self._variable)
                if self._skew > 0:
                    noise = eu.probability.SampleSkewNorm(0, self._noise_stdev *
                            temp, self._skew)
                    val = temp + noise
                else:
                    exec('temp = sys._' + self._variable)
                    noise = np.random.normal(0, self._noise_stdev * temp)
                    val = temp + noise
            else:
                exec('temp = sys._' + self._variable)
                if self._skew > 0:
                    noise = eu.probability.SampleSkewNorm(0, self._noise_stdev,
                    self._skew)
                    val = temp + noise
                else:                    
                    val = temp + np.random.normal(0, self._noise_stdev)
                                                          
                                                           
            if val > self._range[1] or val < self._range[0]:
                return 'out of range'
            else:
                return val


class LotkaVolterra2DSensor(VABSensor):
    def __init__(self, variable, dynamic_range, noise_stdev=0, proportional=False,
                 skew=0):
        if not variable in set([1, 2]):
            raise ValueError('Inorrect variable specification. Must be 1 or 2.')
        else:
            self._variable = variable
        self._range = dynamic_range
        self._noise_stdev = noise_stdev
        self._proportional = proportional
        self._skew = skew

    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            if self._noise_stdev == 0:
                if self._variable == 1:
                    val = sys._x1
                elif self._varibale == 2:
                    val = sys._x2
                else:
                    raise ValueError('Inorrect variable specification. Must be 1 or 2.')

            elif self._proportional:
                if self._variable == 1:
                    temp = sys._x1
                elif self._variable ==2:
                    temp = sys._x2
                else:
                    raise ValueError('Inorrect variable specification. Must be 1 or 2.')

                if self._skew > 0:
                    s = self._noise_stdev * x / np.sqrt(1 - (2. * self._skew**2) /
                            (np.pi * (1. - self._skew**2)))
                    noise = eu.probability.SampleSkewNorm(0, s, self._skew)
                    val = temp + noise
                else:
                    noise = np.random.normal(0, self._noise_stdev * temp)
                    val = temp + noise
            else:
                if self._variable == 1:
                    temp = sys._x1
                elif self._variable ==2:
                    temp = sys._x2
                else:
                    raise ValueError('Inorrect variable specification. Must be 1 or 2.')

                if self._skew > 0:
                    s = self._noise_stdev / np.sqrt(1 - (2. * self._skew**2) /
                            (np.pi * (1. - self._skew**2)))
                    noise = eu.probability.SampleSkewNorm(0, s, self._skew)
                    val = temp + noise
                else:                    
                    val = temp + np.random.normal(0, self._noise_stdev)
                                                          
                                                           
            if val > self._range[1] or val < self._range[0]:
                return 'out of range'
            else:
                return val
                

class LotkaVolterraNDSensor(VABSensor):
    def __init__(self, variable, dynamic_range, noise_stdev=0, proportional=False,
                 skew=0):

        self._variable = variable
        self._range = dynamic_range
        self._noise_stdev = noise_stdev
        self._proportional = proportional
        self._skew = skew

    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            if self._noise_stdev == 0:
                val = 0
                val = sys._x[self._variable-1]

            elif self._proportional:
                temp = 0
                temp = sys._x[self._variable-1]


                if self._skew > 0:
                    s = self._noise_stdev * x / np.sqrt(1 - (2. * self._skew**2) /
                            (np.pi * (1. - self._skew**2)))
                    noise = eu.probability.SampleSkewNorm(0, s, self._skew)
                    val = temp + noise
                else:
                    noise = np.random.normal(0, self._noise_stdev * temp)
                    val = temp + noise
            else:
                temp = sys._x[self._variable-1]

                if self._skew > 0:
                    s = self._noise_stdev / np.sqrt(1 - (2. * self._skew**2) /
                            (np.pi * (1. - self._skew**2)))
                    noise = eu.probability.SampleSkewNorm(0, s, self._skew)
                    val = temp + noise
                else:                    
                    val = temp + np.random.normal(0, self._noise_stdev)
                                                          
                                                           
            if val > self._range[1] or val < self._range[0]:
                return 'out of range'
            else:
                return val

