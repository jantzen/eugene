import math
import time
import copy
import random
import numpy as np

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
        of an index variable, and one n-column array for each of one or more 
        distinct initial values of a n-tuple of target variables.
    """
    def __init__(self, index_id = None, index_values =
        np.array([],dtype='float64'), target_ids = [], target_values =
        []):
        # store the id of the index variable
        if index_id != None and type(index_id) == int:
            self._index_id = index_id
        else:
            raise ValueError("id of index variable must be an integer")

        # store the ids of the target variables
        self._target_ids = target_ids
        
        # store passed data
        self._index_values = index_values
        self._target_values = target_values

    def SetIndex(self, index_id):
        if type(index_id) == int:
            self._index_id = index_id
        else:
            raise ValueError("id must be an integer")

    def SetTarget(self, target_ids):
        self._target_id = target_id

    def SetIndexData(self, data_array):
        self._index_values = data_array

    def SetTargetData(self, list_of_data_arrays):
        self._target_values = list_of_data_arrays



##################################################################
##################################################################
#Functions:

#TimeSampleData requires: class DataFrame
def TimeSampleData(time_var, target_vars, interface, ROI, resolution=[100,10],
        target_value_points=False):
    """ Fills and returns a DataFrame object with time-sampled data from the
    specified system.

    ROI: a dictionary whose key values are ids for variables and whose entries
    are lists indicating a range (i.e., [low, high]) for each variable

    target_value_points: This is a flag indicating whether or not target
    variable ROIs should be interpreted as regions or fixed sets of initial values
    """
    # figure the delay to set between samples
    [time_low, time_high] = ROI[time_var]
    delay = (float(time_high) - float(time_low)) / resolution[0]

    # fill out the index variable (time) values for the data frame
    times = np.arange(time_low, time_high+delay, delay)
    times = times.reshape(len(times),1) # reshape as column
    
    # figure out the set of initial values for each target variable to be
    # sampled
    if target_value_points == False:
        initial_values = []
        for var in target_vars:
            [target_low, target_high] = ROI[var]
            spacing = (float(target_high) - float(target_low)) / resolution[1]
            initial_values.append(np.arange(target_low, target_high, spacing))
    else:
        initial_values = []
        for var in target_vars:
            initial_values.append(ROI[var])

    samples = []
    num_initial_vals = len(initial_values[0]) 
    # (each list in initial_values should have the same length)


    # for each target variable and each initial value, sample data
    for i in range(num_initial_vals):
        for j, var in enumerate(target_vars):
            # reset the system to t=0, then increment to time_low
            interface.set_actuator(var, initial_values[j][i])
        interface.set_actuator(time_var, time_low)
    
        # initialize the output list
        data_list = [[] for i in range(len(target_vars))]

        # start sampling
        for t in range(len(times)):

            for pos, var in enumerate(target_vars):
                data_list[pos].append(interface.read_sensor(var))

            interface.set_actuator(time_var, delay)
    
        # convert the data and add to output list
        num_points = len(data_list[0])
        for num_col, col in enumerate(data_list):
            data_list[num_col] = np.array(col).reshape([num_points, 1])

        data_block = np.concatenate(data_list, 1)
        samples.append(data_block)

    out = DataFrame(time_var, times, target_vars, samples)
    return out
