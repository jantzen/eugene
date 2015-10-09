import math
import time
import copy
import random
import numpy as np
#import sensors & actuators
import connect





class VABSystemInterface( object ):
    """ This is a generic class. Interface objects are what the 
    'process' will operate on directly.
    """

    def __init__(self, sensors, actuators, system = None):
    # sensors: a dictionary of sensor objects, 
    # actuators: is a dictionary of actuator objects, 
    # system is either null (for real physical systems) or a
        # system object for theoretical systems.
    
        self._system = system
        self._sensors = sensors
        self._actuators = actuators


    def read_sensor(self, id):
        # pull the correct sensor
        s = self._sensors[id]
        # test whether system is null
        if self._system == None:
            return s.read()
        else:
            return s.read(self._system)


    def set_actuator(self, id, value):
        # pull the correct actuator
        a = self._actuators[id]
        # test whether system is null
        if self._system == None:
            a.set(value)
        else:
            a.set(self._system, value)


    def get_sensor_range(self, id):
        # return the dynamic range of the sensor corresponding to id
        return self._sensors[id].get_range()
        
    def reset(self):
        self._system.reset()


