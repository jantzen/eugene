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

