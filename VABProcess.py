# VABProcess.py

import time
import math
import random

def SameState(state1, state2, tolerance):
    #DEPRECATED
    if abs(state1 - state2) < tolerance:
        return True
    else:
        return False


def SymFunc(interface, func, time_var, intervention_var, inductive_threshold, time_interval):
    """ Takes a system interface, a function (explicit, no free params)
    representing the intervention being tested for symmetry status, a number
    indicating the variable on which the function is presumed to act, a tolerance (how close
    states need to be in order to count as the same), and an inductive threshold
    (how many affirmative instances are required in order to generalize), and a
    time interval which determines how long to evolve a system and look for
    differences. Returns a logical truth value.
    """

    # EVOLVE AND THEN TRANSFORM
    
    # record the initial state (so it can be replicated in the next step)
    t0 = interface.read_sensor(time_var)
    v0 = interface.read_sensor(intervention_var)
    
    # evolve the system
    time.sleep(time_interval)

    # transform the system
    interface.set_actuator(intervention_var,
            func.evaluateAt(interface.read_sensor(intervention_var)))

    # immediately read the new state of affairs
    v1 = interface.read_sensor(intervention_var)

    # TRANSFORM AND THEN EVOLVE

    # get the system back into its initial state and apply the transformation
    # (in a single step)
    interface.set_actuator(intervention_var, func.evaluateAt(v0))

    # evolve the system
    time.sleep(time_interval)

    # read the final state of affairs
    v2 = interface.read_sensor(intervention_var)

    
    # COMPARE THE FINAL STATES
    return abs(v1, v2)/v1

def SymmetryGroup(interface, func, time_var, intervention_var, inductive_threshold, time_interval, const_ranges):
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
        for y in const_ranges:
            constants.append(random.uniform(y._start, y._end))
        func.set_constants(constants)
        
        #Test whether func with the generated constants is a symmetry
        sum += pow(SymFunc(interface, func, time_var, intervention_var, inductive_threshold, time_interval), 2)
        
    return (sum/inductive_threshold)
    

def GeneticAlgorithm(interface, current_generation, generation_average, time_var, intervention_var, inductive_threshold, time_interval, const_ranges, deck, generation_limit, generation):
    """ Genetic Algorithm to find functions which are most likely to be symmetries
    
        Parameters:
            interface: System interface that is used to modify/monitor the system
            current_generation: The current generation of the GA
            generation_average: The average fitness of the last generation
            time_var: The sensor ID of the time sensor
            intervention_var: The sensor ID of the actuator
            inductive_threshold: The number of trials before generalizing
            time_interval: Wait time
            const_ranges: Ranges of possible values for the constants in the functions
            deck: Set of expressions used to mutate functions
            generation_limit: Maximum number of generations before ending
            generation: The current generation
    """
    nextGeneration = []
    fitnessTotal = 0
    
    #Loop through all of the current-gen functions
    for func in current_generation:
        for func2 in deck:
            #Combine the function with the functions in the deck in various ways
            modifiedFuncs = allOperations(func, func2)
            for function in modifiedFuncs:
                #Measure fitness
                function._fitness = SymmetryGroup(interface, function, time_var, intervention_var, inductive_threshold, time_interval, const_ranges)
                if function._fitness > generation_average:
                    nextGeneration.append(function)
    for func in nextGeneration:
        fitnessTotal += func._fitness
    if generation > generation_limit:
        return nextGeneration
    return GeneticAlgorithm(interface, nextGeneration, fitnessTotal/len(nextGeneration), time_var, intervention_var, inductive_threshold, time_interval, const_ranges, deck, generation_limit, generation + 1)
    
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