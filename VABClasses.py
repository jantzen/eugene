# VABClasses.py
""" Set of classes for VAB objects.
"""

import math
import time
import copy
import random


class VABSensor( object ):
    def __init__(self, dynamic_range):
        self._init_value = None
        self._range = dynamic_range

    def get_range(self):
        return self._range


class VABTimeSensor(VABSensor):
    def read(self, sys):
        return sys.update_time()


class VABPopulationSensor(VABSensor):
    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            pop = sys.update_pop()
            if pop > self._range[1] or pop < self._range[0]:
                return 'OutofRange'
            else:
                return pop

class VABLogisticSensor(VABSensor):
    def read(self, sys):
        if len(self._range) != 2:
            raise ValueError('No sensor range specified.')
        else:
            pop = sys.update_x()
            if pop > self._range[1] or pop < self._range[0]:
                return 'OutofRange'
            else:
                return pop

class VABPopulationActuator(VABSensor):
    def set(self, sys, value):
        sys._population = value
        sys._time = time.time()

class VABLogisticActuator(VABSensor):
    def set(self, sys, value):
        sys._x = value
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
        
        self._init_pop = init_pop

    
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
        
    def reset(self):
        self._population - self._init_pop
        
class VABSystemLogistic( object ):

    def __init__(self, malthusian_param, x_init):
        self._malthusian = malthusian_param
        self._x = x_init
        self._time = time.time()
        self._x0 = x_init
        
    def update_x(self):
        curr_time = time.time()
        elapsed_time = curr_time - self._time
        self._x = 1/(1+((1/self._x)-1)*math.exp(0-self._malthusian*elapsed_time))
        self._time = curr_time
        return self._x
    def update_time(self):
        curr_time = time.time()
        elapsed_time = curr_time - self._time
        self._x = 1/(1+((1/self._x)-1)*math.exp(0-self._malthusian*elapsed_time))
        self._time = curr_time
        return self._x
        
    def reset(self):
        self._x = self._x0

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


    def get_sensor_range(self, id):
        # return the dynamic range of the sensor corresponding to id
        return self._sensors[id].get_range()
        
    def reset(self):
        self._system.reset()


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
        self._error = float('Inf')

    def __str__(self):
        return self._function

    def __repr__(self):
        return str(self)

    def SetConstants(self, constants):
        """Sets the constants in the function
        """
        if len(constants) == self._const_count: 
            self._constants = constants
        else:
            raise ValueError
        
    def EvaluateAt(self, v):
        """ Evaluates the function when the variables v1, v2, v3, ...
            are at v[0], v[1], v[2], ...
        """
        c = self._constants

        return eval(self._function)
        
    def Multiply(self, factor_in):
        """Multiplies this function by another
        """
        factor = Function(factor_in._function, factor_in._const_count, factor_in._var_count)
        # This creates a local copy so we don't change the passed function object
        factor.IncrementIndices(self._const_count)
        self._const_count += factor._const_count
        self._function = '(' + self._function + ')*(' + factor._function + ')'
        
    def Add(self, addend_in):
        """Adds this function to another
        """
        addend = Function(addend_in._function, addend_in._const_count, addend_in._var_count)
        # This creates a local copy so we don't change the passed function object
        addend.IncrementIndices(self._const_count)
        self._const_count += addend._const_count
        self._function = '(' + self._function + ')+(' + addend._function + ')'
        
    def Power(self, power_in):
        """Raises the function to a power
        """
        power = Function(power_in._function, power_in._const_count, power_in._var_count)
        # This creates a local copy so we don't change the passed function object
        power.IncrementIndices(self._const_count)
        self._const_count += power._const_count
        self._function = 'pow(' + self._function + ',' + power._function + ')' 
        
    def IncrementIndices(self, numConsts):
        """ Increments the indices of the constants by the amount
            given by the passed parameters
        """
        Digits = '0123456789';
        descriptorIndex = -2
        descriptor = ''
        num = ''
        updatedFunction = ''
        for c in self._function:
            if c in Digits:
                num += c
                if descriptor == '':
                    descriptor = self._function[descriptorIndex]
            else:
                if num != '' and descriptor == 'c':
                    updatedFunction += str((int(num)) + numConsts)
                    num = ''
                    descriptor = ''
                elif num != '' and descriptor == 'v':
                    updatedFunction += str((int(num)))
                    num = ''
                    descriptor = ''
                updatedFunction += c
            descriptorIndex += 1
        self._function = updatedFunction
    
    def Substitute(self, expression, placeholder):
        self._function.replace(placeholder, expression)

class Expression( object ):
    """Recursive structure for holding a mathematical expression.  An expression
    may have 0, 1, or 2 children and an operation that acts upon them (if there is
    at least one) or a symbol
    """
    def __init__(self, terminalSymbol, const_count = 0):
        """
        Initializes an expression that consists only of a terminal symbol
        """
        self._terminal = terminalSymbol
        self._left = None
        self._right = None
        self._const_count = const_count
    
    def SetLeft(self, expression):
        self._left = expression
    
    def SetRight(self, expression):
        self._right = expression
        
    def SetTerminal(self, expression):
        self._terminal = expression
        
    def Evaluate(self):
        """Generates a string representation of the function, with all parameters represented
        as c[0]
        """
        if self._right != None:
            return '(' + self._left.Evaluate() + self._terminal + self._right.Evaluate() + ')'
        elif self._left != None:
            return self._terminal + '(' + self._left.Evaluate() +')'
        else:
            return self._terminal
    
    def Size(self):
        if self._right != None:
            return self._left.Size() + 1 + self._right.Size()
        elif self._left != None:
            return 1 + self._left.Size()
        else:
            return 1
            
    def CountConstants(self):
        if self._right != None:
            return self._left.CountConstants() + self._const_count + self._right.CountConstants()
        elif self._left != None:
            return self._const_count + self._left.CountConstants()
        else:
            return self._const_count
        
class FunctionTree( object ):
    """ Class that represents a function composed of variables (v), parameters
    (c), and operations that act on them.  The function itself is stored as an
    Expression.
    """
    def __init__(self, initialExpression):
        """Initial Expression must be a valid expression
        """
        self._expression = copy.deepcopy(initialExpression)
        self._parameters = [0]
        self._error = 10**12
        
    def EvaluateAt(self, v):
        """Evaluates the function with the variables, v0, v1, etc. equal to
        v[0], v[1], etc.
        """
        c = self._parameters
        func = self.ExpressionString()
        return eval(func)
        
    def ExpressionString(self):
        """Turns the expression into a readable and executeable string with
        parameters properly indexed
        """
        expression = self._expression.Evaluate()
        Digits = '0123456789';
        descriptorIndex = -2
        descriptor = ''
        num = ''
        const_count = 0
        updatedFunction = ''
        for c in expression:
            if c in Digits:
                num += c
                if descriptor == '':
                    descriptor = expression[descriptorIndex]
            else:
                if num != '' and descriptor == 'c':
                    updatedFunction += str(const_count)
                    const_count += 1
                    num = ''
                    descriptor = ''
                elif num != '':
                    updatedFunction += str((int(num)))
                    num = ''
                    descriptor = ''
                updatedFunction += c
            descriptorIndex += 1
        return updatedFunction
    
    def CountParameters(self):
        """Count the number of parameters (c) in the expression
        """
        return self._expression.CountConstants()
    
    def SetParameters(self, parameters):
        """Sets the value for the expression's paremeters
        """
        if len(parameters) == self.CountParameters(): 
            self._parameters = parameters
        else:
            raise ValueError
    
    def Size(self):
        """Computes the size of the expression
        """
        return self._expression.Size()
        
    def OperateOnRandom(self, operation, operand=None):
        """Performs an operation to a random sub-expression of this function
        """
        self.OperateOn(random.randint(1,self.Size()), operation, operand)
    
    def OperateOn(self, node, operation, operand=None, index=0, exp=None):
        """Performs an operation on sub-expression node, where node is an integer 
        indicating the sub-expression's position in an in-order traversal of the tree
        """
        if exp == None:
            self.OperateOn(node, operation, operand, index, self._expression)
        elif index < node:
            if exp._left != None:
                index += self.OperateOn(node, operation, operand, index, exp._left)
            index +=1
            if index == node:
                self.Operate(exp, operation, operand)
            if exp._right != None:
                index += self.OperateOn(node, operation, operand, index, exp._right)
        return index
        
    def Operate(self, expression, operation, operand=None):
        """Performs the given operation on the given function
        """
        left_replica = copy.deepcopy(expression)
        right_replica = copy.deepcopy(operand)
        expression.SetTerminal(operation)
        expression.SetLeft(left_replica)
        expression.SetRight(right_replica)
        expression._const_count = 0
        
        
        
        
        
        
        
        
        
