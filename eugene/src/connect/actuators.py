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
#        sys._time = time.time()

#       what if "system = None"
        sys._time = sys._init_t



# Actuator to kick virtual circuit, based on concentration material       
class VABVoltageActuator(VABSensor):
    def set(self, sys, value):
        sys._v = value
        sys._time = sys._init_t    

class PopulationActuator(VABSensor):
    def set(self, sys, value):
        sys._x = value
        sys._time = sys._init_t

