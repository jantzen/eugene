# VABSystemExpGrowth.py
""" System class for simulation of exponential growth.
"""

import math
import time


class VABSensor( object ):
    def __init__self(self):
        self._init_value = None

class VABTimeSensor(VABSensor):
    def read(self, sys):
        return sys.update_time()


class VABPopulationSensor(VABSensor):
    def read(self, sys):
      return sys.update_pop() 


class VABPopulationActuator(VABSensor):
    def set(self, sys, value):
        sys._population = value
        sys._time = time.time()


class VABSystemExpGrowth(object):
    """This class defines a simulated system -- a collection of 
    variables and the functions that relate them in structural
    equation form.
    """
    
    def __init__(self, init_pop, growth_rate):
        # set the initial population based on passed data
        if init_pop > 0:
            self._population = init_pop
        else:
            raise ValueError('Invalid initial population assignment. Must be greater than 0')

        # set the time corresponding to that initial population
        self._time = time.time()

        #set remaining attributes
        self._growth_rate = growth_rate

    
    """ define a function that returns the current population given
    the population at some earlier time, time_i.
    """
    def update_pop(self):
        # first get current time
        curr_time = time.time()
        # compute the time elapsed since the last read
        elapsed_time = curr_time - self._time
        # now compute the current population based on the model
        pop = self._population*math.exp(self._growth_rate*elapsed_time)
        # update the class variables before returning the values
        self._population = pop
        self._time = curr_time
        
        return self._population

    def update_time(self):
        # first get current time
        curr_time = time.time()
        # compute the time elapsed since the last read
        elapsed_time = curr_time - self._time
        # now compute the current population based on the model
        pop = self._population*math.exp(self._growth_rate*elapsed_time)
        # update the class variables before returning the values
        self._population = pop
        self._time = curr_time
        
        return self._time

class VABSystemInterface( object ):
    """ This is a generic class. Interface objects are what the 'process' will
    operate on directly.
    """

    def __init__(self, sensors, actuators, system = None):
    """ sensors is a dictionary of sensor objects, actuators is a dictionary of
    actuator objects, system is either null (for real physical systems) or a
    system object for theoretica systems
    """

        self._system = system


    def read_sensor(id):
        # pull the correct sensor
        s = sensors[id]
        # test whether system is null
        if system == None:
            return s.read()
        else:
            return s.read(self._system)


    def set_actuator(id, value):
        # pull the correct actuator
        a = actuators[id]
        # test whether system is null
        if system == None:
            a.set(value)
        else:
            a.set(self._system, value)

