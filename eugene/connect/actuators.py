import math
import time
import copy
import random
import numpy as np

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

#Real Actuators
class VABConcentrationActuator(VABSensor):
    def set(self, sys, value):
        sys._x = value
#        sys._time = time.time()

#       what if "system = None"
        sys._time = sys._init_t

