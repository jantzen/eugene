# VABProcess.py

import time
import math
import random
from VABClasses import *

def SameState(state1, state2, tolerance):
    #DEPRECATED
    if abs(state1 - state2) < tolerance:
        return True
    else:
        return False


def SymFunc(interface, func, time_var, intervention_var, inductive_threshold, time_interval):
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

    
    # COMPARE THE FINAL STATES
    return abs(v1 -  v2)/v1

    # NOTE: DOES NOT USE THE INDUCTIVE THRESHOLD


def SymmetryGroup(interface, func, time_var, intervention_var, inductive_threshold, time_interval, const_range):
    """ Tests to see if several variations of func are symmetries.  
        If every variation of constants is a symmetry,returns true.
        This is the fitness function for the GA
        const_ranges: A list containing the ranges in which each constant in the function must reside
        trials: The number of different random variations of constants to test
    """ 
    sum = 0
    for x in range(0, inductive_threshold):
        constants = [];
        
        #Generate a list of constants
        for y in range(0,func._const_count):
            constants.append(random.uniform(const_range._start, const_range._end))
        func.SetConstants(constants)
        
        #Test whether func with the generated constants is a symmetry
        sum += pow(SymFunc(interface, func, time_var, intervention_var, inductive_threshold, time_interval), 2)
        
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
        #Loop through all of the current-gen functions
        for func in current_generation:
            for func2 in RandomSelection(deck, num_mutes):
                #Combine the function with the functions in the deck in various ways
                modifiedFunc = randomOperation(func, func2)
                #Measure fitness
                modifiedFunc._fitness = SymmetryGroup(interface, modifiedFunc, time_var, intervention_var, inductive_threshold, time_interval, const_range)
                nextGeneration.append(modifiedFunc)
                        
        #Remove the functions that will not be passed on to the next generation:
                        
        #Sort the next generation by fitness
        comparison = lambda x: x._fitness
        nextGeneration.sort(comparison, reverse=True)
        if generation_size > percent_guaranteed * len(nextGeneration):
            fitnessTotals = []
            
            #Preserve the most fit functions
            reducedGeneration = nextGeneration[0:percent_guaranteed*len(nextGeneration)]
            runningTotal = 0
            
            #Create a list of running totals, representing a weighted distribution
            for func in nextGeneration:
                runningTotal += func._fitness
                fitnessTotals.append(runningTotal)
                
            #Fill the rest of the spots in the next generation
            for i in range(0, generation_size - percent_guaranteed * len(nextGeneration)):
                
                #Choose a random number that is greater than the running total from the last of the guaranteed-to-pass-on functions
                rand = random.random() * (fitnessTotals[len(fitnessTotals) - 1] - fitnessTotals[percent_guaranteed*len(fitnessTotals)])
                
                #Use the random number and the weighted distribution to select a function to preserve
                for j in range(percent_guaranteed*len(nextGeneration), len(nextGeneration)):
                    if fitnessTotals[i] > rand:
                        reducedGeneration.append(fitnessTotals[i])
                        fitnessTotals.remove(fitnessTotals[i])
                        nextGeneration.remove(nextGeneration[i])
            current_generation = reducedGeneration
        else:
            current_generation = nextGeneration
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
                    function._fitness = SymmetryGroup(interface, function, time_var, intervention_var, inductive_threshold, time_interval, const_ranges)
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
<<<<<<< HEAD
    """Returns the result of one of the possible combinations of functions 1 and 2
    chosen at random (from a uniform distribution)
=======
    """Returns the result of a possible combination of functions 1 and 2
>>>>>>> 95fe62fb8f4393dfc83f8e77ff2ca52f32ae1346
    """
    operation = random.randint(0, 2)
    function1Copy = Function(function1._function, function1._const_count, function1._var_count)
    if operation == 0:
<<<<<<< HEAD
        function1.Add(function2)
    elif operation == 2:
        function1.Multiply(function2)
    else:
        function1.Exponentiate(function2)
=======
        function1Copy.Add(function2)
        return function1Copy
    elif operation == 2:
        function1Copy.Multiply(function2)
        return function1Copy
    else:
        function1Copy.Power(function2)
        return function1Copy
>>>>>>> 95fe62fb8f4393dfc83f8e77ff2ca52f32ae1346


def RandomSelection(deck, num_selected):
    """Returns num_selected number of elements of the list, deck
    """
    random.shuffle(deck)
    return deck[0:(num_selected-1)]
    
