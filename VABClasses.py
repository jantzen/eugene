# VABClasses.py
""" Set of classes for VAB objects.
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

class VABSigmoidSystem( object ):
    """ A system that simulates the logistic equation
    """
    def __init__(self, limit, midpoint, steepness):
        self._limit = limit
        self._midpoint = midpoint
        self._steepness = steepness
        
    def update(self, x):
        return self._limit/(1+pow(math.e, 0-(self.steepness*(x - self._midpoint))))
        

class VABSystemInterface( object ):
    """ This is a generic class. Interface objects are what the 'process' will
    operate on directly.
    """

    def __init__(self, sensors, actuators, system = None):
    # sensors is a dictionary of sensor objects, actuators is a dictionary of
    # actuator objects, system is either null (for real physical systems) or a
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

    
class Range( object ):
    """Specifies both the start and the end of a range of numbers
    """
    def __init__(self, start, end):
        self._start = start
        self._end = end
        
        
class Function( object ):
    """ Contains a function that can be added, multiplied by, and raised to the
        power of another function.
        The function is stored as a string.  The function f(x) = kx would be
        stored as the string "c[0]x
    """
    def __init__(self, func, const_count, var_count):
        self._function = func
        self._const_count = const_count
        self._var_count = var_count
        self._constants = [0] * const_count
        
    def SetConstants(self, constants):
        """Sets the constants in the function
        """
        self._constants = constants
        
    def EvaluateAt(self, v):
        """ Evaluates the function when the variables v1, v2, v3, ...
            are at v[0], v[1], v[2], ...
        """
        c = self._constants
        return eval(self._function)
        
    def Multiply(self, factor):
        """Multiplies this function by another
        """
        factor.IncrementConstants(self._const_count, self._var_count)
        self._const_count += factor._const_count
        self._function = '(' + self._function + ') * (' + factor._function + ')'
        
    def Add(self, addend):
        """Adds this function to another
        """
        
        addend.IncrementConstants(self._const_count, self._var_count)
        self._const_count += addend._const_count
        self._function = '(' + self._function + ') + (' + addend._function + ')'
        
    def Power(self, power):
        """Raises the function to a power
        """
        
        power.IncrementConstants(self._const_count, self._var_count)
        self._const_count += power._const_count
        self._function = 'pow(' + self._function + ', ' + power._function + ')' 
        
    def IncrementConstantsAndVariables(self, numConsts, numVars):
        """ Increments the indexes of the constants and variables by the amount
            given by the parameters
        """
        Digits = '0123456789';
        descriptorIndex = -2
        num = ''
        updatedFunction = ''
        for c in self._function:
            if c in Digits:
                num += c
            else:
                if num != '' and self._function[descriptorIndex] == 'c':
                    updatedFunction += ((int(num)) + numConsts)
                    num = ''
                elif num != '' and self._function[descriptorIndex] == 'v':
                    updatedFunction += ((int(num)) + numVars)
                    num = ''
                updatedFunction += c
            descriptorIndex += 1
        self._function = updatedFunction
    
    def IncrementConstants(self, numConsts):
        """ Increments the indexes of the constants by the amount
            given by the parameters
        """
        Digits = '0123456789';
        descriptorIndex = -2
        num = ''
        updatedFunction = ''
        for c in self._function:
            if c in Digits:
                num += c
            else:
                if num != '' and self._function[descriptorIndex] == 'c':
                    updatedFunction += ((int(num)) + numConsts)
                    num = ''
                updatedFunction += c
            descriptorIndex += 1
        self._function = updatedFunction
    
    def Substitute(self, expression, placeholder):
        self._function.replace(placeholder, expression)