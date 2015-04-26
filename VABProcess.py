# VABProcess.py

import time
import math
import random
#import scipy
from VABClasses import *
from scipy import optimize


def SameState(state1, state2, tolerance):
    #DEPRECATED
    if abs(state1 - state2) < tolerance:
        return True
    else:
        return False


def FindConstants(interface, func, time_var, intervention_var, time_interval, const_range):
    good_set = False
    lower = interface.get_sensor_range(intervention_var)[0]
    upper = interface.get_sensor_range(intervention_var)[1]
    attempts = 0

    while not good_set and attempts<10:
        #Try/catch added to take care of cases such as exp(exp(v[0])), which are likely not desirable and almost certainly lead to overflows
        try:
            # Generate a tentative list of constants
            constants = [] 
            #for y in range(0,func._const_count):
            for y in range(0,func.CountParameters()):
                constants.append(random.uniform(const_range._start, const_range._end))
            func.SetParameters(constants)
            
            # Check whether the transformation, given these parameter values, would keep the intervention_var in of bounds
            val1 = func.EvaluateAt([interface.read_sensor(intervention_var)])

            if  val1 > lower and val1 < upper:
                # Check whether the transformation would take the system out of bounds assuming that time evolution takes it half way there (in either direction)
                temp1 = (val1 + upper)/2 
                temp2 = (val1 + lower)/2
    
                val2_high = func.EvaluateAt([temp1])
                val2_low = func.EvaluateAt([temp2])
                if val2_high < upper and val2_low > lower:
                    good_set = True
            attempts += 1
        except OverflowError:
            attempts += 1
            print 'Overflow from ' + func.ExpressionString()
        
    if good_set:
       return 1

    else:
        return 0


def SymFunc(interface, func, time_var, intervention_var, time_interval):
    """ Takes a system interface, a function object...
        FINISH THIS DESCRIPTION 
    """

    # EVOLVE AND THEN TRANSFORM
    
    # record the initial state (so it can be replicated in the next step)
    t0 = interface.read_sensor(time_var)
    v0 = interface.read_sensor(intervention_var)
    
    # evolve the system
    time.sleep(time_interval)

    # transform the system
    interface.set_actuator(intervention_var,
            func.EvaluateAt([interface.read_sensor(intervention_var)]))

    # immediately read the new state of affairs
    v1 = interface.read_sensor(intervention_var)


    # TRANSFORM AND THEN EVOLVE

    # get the system back into its initial state and apply the transformation
    # (in a single step)
    interface.set_actuator(intervention_var, func.EvaluateAt([v0]))

    # evolve the system
    time.sleep(time_interval)

    # read the final state of affairs
    v2 = interface.read_sensor(intervention_var)

    #DEBUGGING:
    print "SymFunc output: function = {}, v1 = {}, v2 = {}, v1-v2 = {}".format(func._expression.Evaluate(),v1,v2,v1-v2)

    # COMPARE THE FINAL STATES
    return (v1 -  v2)

    

def SymmetryGroup(interface, func, time_var, intervention_var, inductive_threshold, time_interval, const_range):
    """ Tests to see if several variations of func are symmetries.  
        Returns the mean of squares of the error
        This is the fitness function for the GA
        const_ranges: A list containing the ranges in which each constant in the function must reside
        trials: The number of different random variations of constants to test
    """ 
    sum = 0
    for x in range(0, inductive_threshold):
        constants = [];
        
        #Generate a list of constants
        #for y in range(0,func._const_count):
        #    constants.append(random.uniform(const_range._start, const_range._end))
        #func.SetConstants(constants)
        check = FindConstants(interface, func, time_var, intervention_var, time_interval, const_range)   
        if check == 1:
            #Test whether func with the generated constants is a symmetry
            sum += pow(SymFunc(interface, func, time_var, intervention_var, time_interval), 2)
        else:
            sum += 10**12
            print func._expression.Evaluate() + ' could not find constants'
        
    return (sum/inductive_threshold)
    

def GeneticAlgorithm(interface, current_generation, time_var, intervention_var, inductive_threshold, time_interval, const_range, deck, generation_limit, num_mutes, generation_size, percent_guaranteed):
    """ Genetic Algorithm to find functions which are most likely to be symmetries
    
        Parameters:
            interface: System interface that is used to modify/monitor the system
            current_generation: The current generation of the GA
            time_var: The sensor ID of the time sensor
            intervention_var: The sensor ID of the actuator
            inductive_threshold: The number of trials before generalizing
            time_interval: Wait time
            const_ranges: Ranges of possible values for the constants in the functions
            deck: Set of expressions used to mutate functions
            generation_limit: Maximum number of generations before ending
            num_mutes: The number of mutations to apply to each member of the current generation
            generation_size: The maximum allowable size of a single generation
            percent_guaranteed: The top x% of fit functions are guaranteed to be passed to the next generation
    """
    
    for generation in range(0, generation_limit):
        nextGeneration = []
        #All members of the current generation are candidates for the next generation
        nextGeneration.extend(current_generation)
        
        #Loop through all of the current-gen functions
        for func in current_generation:
            for func2 in RandomSelection(deck, num_mutes):
                #Combine the function with the functions in the deck in various ways
                modifiedFunc = randomTreeOperation(func, func2)
                #Measure fitness (convert errors to fitnesses)
                modifiedFunc._error = (SymmetryGroup(interface, modifiedFunc, time_var, intervention_var, inductive_threshold, time_interval, const_range))
                nextGeneration.append(modifiedFunc)
                interface.reset()
             
        #Sort the next generation by fitness 
        comparator = lambda x: x._error
        nextGeneration.sort(key=comparator)
       
        #Figure out how many are guaranteed to be passed on
        numGuaranteed = int(percent_guaranteed * len(nextGeneration))

        #Check to see if we need to save more than the guaranteed percentage
        if generation_size > numGuaranteed:
            #Check whether there's room to save everything
            if generation_size > len(nextGeneration):
                #DEBUGGING
                print "Function with smallest error: {}. Error: {}".format(current_generation[0]._expression.Evaluate(), current_generation[0]._error)
                
                current_generation = nextGeneration
                continue
               
            #We can't save everything.
            #First, preserve the most fit functions
            if numGuaranteed > 0:
                reducedGeneration = nextGeneration[0:numGuaranteed-1]
                nextGeneration = nextGeneration[numGuaranteed : len(nextGeneration)-1]
            
            #Create a list of running totals, representing a weighted distribution
            errorTotals = []
            runningTotal = 0
            for func in nextGeneration:
                runningTotal += func._error
                errorTotals.append(runningTotal)

            #Determine the number of slots to fill
            slots_remaining = generation_size - numGuaranteed

            #Fill the rest of the spots in the next generation
            for i in range(slots_remaining-1):
                #Pull a random number from a uniform distribution on [0,1]
                rand = random.random()

                #Assuming an exponential probability density function over error of the form exp(-x), find the corresponding error value (x)
                x = -math.log(rand)
                
                #Use the generated error value to select a function to preserve
                for j in range(len(nextGeneration)):
                    if errorTotals[j] > x:
                        reducedGeneration.append(nextGeneration[j])
                        errorTotals.remove(errorTotals[j])
                        nextGeneration.remove(nextGeneration[j])
                        break
                        
            current_generation = reducedGeneration
        
        else:
            current_generation = nextGeneration[0:(generation_size-1)]
        
        #DEBUGGING
        print "Lowest error function: {}. Mean Squared Error: {}.".format(current_generation[0].ExpressionString(), current_generation[0]._error)
        print "Generation: {}.".format(generation)

    return current_generation
        
    
def BranchAndBound(interface, seed_func, time_var, intervention_var, inductive_threshold, time_interval, const_ranges, deck, complexity_limit):
    """ Exaustive search of possible functions generated with expressions from 
        the deck
    
        Parameters:
            interface: System interface that is used to modify/monitor the system
            seed_func: The initial function
            time_var: The sensor ID of the time sensor
            intervention_var: The sensor ID of the actuator
            inductive_threshold: The number of trials before generalizing
            time_interval: Wait time
            const_ranges: Ranges of possible values for the constants in the functions
            deck: Set of expressions used to mutate functions
            complexity_limit: Maximum number of expressions before ending
    """
    #All generated functions are stored in possibleSymmetries
    possibleSymmetries = [seed_func]
    for x in range(seed_func._const_count, complexity_limit):
        for func in possibleSymmetries:
            for func2 in deck:
                modifiedFuncs = allOperations(func, func2)
                for function in modifiedFuncs:
                    function._error = SymmetryGroup(interface, function, time_var, intervention_var, inductive_threshold, time_interval, const_ranges)
                possibleSymmetries.append(modifiedFuncs)
    return possibleSymmetries
    
      
def allOperations(function1, function2):
    """Returns the result of each possible combination of functions 1 and 2
    """
    added = function1
    added.Add(function2)
    multiplied = function1
    multiplied.Multiply(function2)
    exponentiated = function1
    exponentiated.Power(function2)
    return [added, multiplied, exponentiated]


def randomOperation(function1, function2):
    """Returns the result of one of the possible combinations of functions 1 and 2
    chosen at random (from a uniform distribution)
    """
    operation = random.randint(0, 1)
    function1Copy = Function(function1._function, function1._const_count, function1._var_count)
    if operation == 0:
        function1Copy.Add(function2)
        return function1Copy
    elif operation == 1:
        function1Copy.Multiply(function2)
        return function1Copy
    #else:
    #    function1Copy.Power(function2)
    #    return function1Copy

def randomTreeOperation(function1, function2):
    """Returns the result of a random combination of function trees 1 and 2
    """
    operations = ['*', '+', 'math.exp', '1/']
    no_param = 2
    opCode = random.randint(0,len(operations)-1)
    replica = copy.deepcopy(function1)
    if opCode >= no_param:
        replica.OperateOnRandom(operations[opCode])
    else:
        print ' ' + str(opCode) + function2.Evaluate()
        replica.OperateOnRandom(operations[opCode], function2)
    return replica
    

def RandomSelection(deck, num_selected):
    """Returns num_selected number of elements of the list, deck
    """
    random.shuffle(deck)
    return deck[0:(num_selected)]
