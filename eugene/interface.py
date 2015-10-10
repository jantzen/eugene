import math
import time
import copy
import random
import numpy as np
#import sensors & actuators
from connect.sensors import *
from connect.actuators import *

#Classes:
# VABSystemInterface
# DataFrame

#Functions:
# TimeSampleData 



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

class DataFrame( object ):
    """ DataFrame objects hold single samples of data suitable for 
        building models of symmetry structure. Specifically, they 
        consist a set of numpy arrays that contains an array of values
        of an index variable, and one array for each of one or more 
        distinct initial values of a target variable.
    """
    def __init__(self, index_id = None, index_values =
    np.array([],dtype='float64'),
            target_id = None, target_values = []):
        # store the id of the index variable
        if index_id != None and type(index_id) == int:
            self._index_id = index_id
        else:
            raise ValueError("id of index variable must be an integer")

        # store the id of the target variable
        if target_id != None and type(target_id) == int:
            self._target_id = target_id
        else:
            raise ValueError("id of target variable must be an integer")

        # store passed data
        self._index_values = index_values
        self._target_values = target_values

    def SetIndex(self, index_id):
        if type(index_id) == int:
            self._index_id = index_id
        else:
            raise ValueError("id must be an integer")

    def SetTarget(self, target_id):
        if type(target_id) == int:
            self._target_id = target_id
        else:
            raise ValueError("id must be an integer")

    def SetIndexData(self, data_array):
        self._index_values = data_array

    def SetTargetData(self, list_of_data_arrays):
        self._target_values = list_of_data_arrays



##################################################################
##################################################################
#Functions:

#TimeSampleData requires: class DataFrame
def TimeSampleData(time_var, target_var, interface, ROI, resolution=[100,10]):
    """ Fills and returns a DataFrame object with time-sampled data from the
    specified system.
    """
    # figure the delay to set between samples
    [time_low, time_high] = ROI[time_var]
    delay = (float(time_high) - float(time_low)) / resolution[0]

    # fill out the index variable (time) values for the data frame
    times = np.arange(time_low, time_high+delay, delay)
    
    # figure out the set of initial values
    [target_low, target_high] = ROI[target_var]
    spacing = (float(target_high) - float(target_low)) / resolution[1]
    initial_values = np.arange(target_low, target_high, spacing)

    samples = []

    # for each intial value, sample data
    for val in initial_values:
        # reset the system to t=0, then increment to time_low
        interface.reset()
        interface.set_actuator(target_var, val)
        interface.set_actuator(time_var, time_low)
    
        # initialize the output list
        data = []
    
        # start sampling
        for i in range(len(times)):
            data.append(interface.read_sensor(target_var))
            interface.set_actuator(time_var, delay)
    
        # convert the data and add to output list
        data = np.array(data)
        samples.append(data)

    out = DataFrame(time_var, times, target_var, samples)

    return out


