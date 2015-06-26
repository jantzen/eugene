# VABProcess.py

import time
import math
import random
#import scipy
#import numpy as np
import numpy
from VABClasses import *
#from scipy import optimize
import pdb
import pp


def SameState(state1, state2, tolerance):
    #DEPRECATED
    if abs(state1 - state2) < tolerance:
        return True
    else:
        return False

def EmpiricalDeriv(data, h=1):
    """ Uses the method of Lanczos (_Applied Analysis_, p321) to
        compute the first derivative of an empirical function.
        data is a list and is assumed to contain an odd number of
        ordinates. The center value is treated as the ordinate 
        corresponding to the abscissa at which the derivative is
        desired. 'h' is the distance along the abscissa between
        samples.
    """
    if len(data)%2 != 0 and len(data)<5:
        raise ValueError('data list must have len >= 5 and contain an odd number of points.')
    if h<=0:
        raise ValueError('step size, h, must be greater than zero')

    k = (len(data)-1)/2
    numerator = 0
    denominator = 0

    index = 0
    for a in range(-k,(k+1)):
        numerator += a*data[index]
        index += 1
    
    for a in range(1,(k+1)):
        denominator += 2*a**2*h

    return numerator/denominator


def DynamicRange(interface, target_var, index_var):
    """ Returns the minimum and maximum values that target_var can
        reach as a function of index_var, bounded by the sensor
        limits for the system. This version assumes that index_var
        is time.
    """
    # Compute the first derivative near the initial value of the system
    # For now, assume the index variable is time. This should
    # be changed in future versions of this package.
    interface.reset()

    # Sample 5 points
    x = []
    for i in range(5):
        x.append(interface.read_sensor(target_var))
        time.wait(0.05)

    # Compute the derivative
    d1 = EmpiricalDeriv(x,0.05)

    # Put the system near the sensor limit (assume this doesn't break causal structure)
    max_x = interface.get_sensor_range(target_var)[1]
    interface.set_actuator(target_var,0.9*max_x)
    
    # Sample 5 points
    x = []
    try:
        for i in range(5):
            x.append(interface.read_sensor(target_var))
            time.wait(0.05)

        # Compute the derivatice
        d2 = EmpiricalDeriv(x,0.05)
    except:
        # If the sensor threw an out-of-range error, the system must increase at that point,
        # so that's the global max.
        global_max = interface.get_sensor_range(target_var)[1]
        


def FindParamVals(interface, func, time_var, intervention_var, time_interval, const_range):
#    good_set = False
#    lower = interface.get_sensor_range(intervention_var)[0]
#    upper = interface.get_sensor_range(intervention_var)[1]
#    attempts = 0

#    while not good_set and attempts<10:
#        #Try/catch added to take care of cases such as exp(exp(v[0])), which are likely not desirable and almost certainly lead to overflows
#        try:
#            # Generate a tentative list of constants
#            constants = [] 
#            #for y in range(0,func._const_count):
#            for y in range(0,func.CountParameters()):
#                constants.append(random.uniform(const_range._start, const_range._end))
#            func.SetParameters(constants)
#            
#            # Check whether the transformation, given these parameter values, would keep the intervention_var in bounds 
#            current_val = interface.read_sensor(intervention_var)
#            try:
#                val1 = func.EvaluateAt([current_val])
#            except:
#                attempts+=1
#                continue
#
#            if  val1 > lower and val1 < upper:
#                # Check whether the transformation would take the system out of bounds assuming that time evolution takes it half way there (in either direction)
#                temp1 = (val1 + upper)/2 
#                temp2 = (val1 + lower)/2
#    
#                val2_high = func.EvaluateAt([temp1])
#                val2_low = func.EvaluateAt([temp2])
#                if val2_high < upper and val2_low > lower:
#                    good_set = True
#            attempts += 1
#        except OverflowError:
#            attempts += 1
#            print 'Overflow from ' + func.ExpressionString()
#        
#    if good_set:
#       return 1
#
#    else:
#        return 0
    from scipy import optimize
    tol = 10**(-6)
#    # read the current state of the system
#    init_val = interface.read_sensor(intervention_var)
#    # allow the system to evolve over a short interval
#    time.sleep(time_interval)
#    # read the evolved state of the system
#    final_val = interface.read_sensor(intervention_var)
#
#    pdb.set_trace()
    # get the range of possible values the system can take (assuming they span the sensor
    # range)
    [f_min,f_max]=interface.get_sensor_range(intervention_var)

    # choose a value in this range at random
    f_target = random.uniform(1.1*f_min,0.9*f_max)
   
   # read the current state of the system
    init_val = interface.read_sensor(intervention_var)
    
    # define functions that will allow a search for appropriate constants
    f = lambda c,v: eval(func.ExpressionString())
    f_min = lambda c: (f(c, [init_val]) - f_target)**2

    # construct an initial guess (an array of 1s of length equal to the number of params)
    init_guess = numpy.array([1] * func.CountParameters())
    try: 
        sol = optimize.minimize(f_min, init_guess, tol=tol)
    except:
        return 0
    
    # if successful, set the parameters in func, reset the system, and return
    if sol.success == True and sol.fun < tol:
        func.SetParameters(sol.x.tolist())
        interface.set_actuator(intervention_var, init_val)
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
    # time.sleep(time_interval)
    interface.set_actuator(time_var, time_interval)

    # transform the system

    # before attempting the transformation, make sure the function evaluates
    # at the desired point (this is going to introduce more noise because
    # the system is going to continue to evolve.
    try:
        set_point = func.EvaluateAt([interface.read_sensor(intervention_var)])
        interface.set_actuator(intervention_var, set_point)
    # interface.set_actuator(intervention_var,
    #        func.EvaluateAt([interface.read_sensor(intervention_var)]))

    # immediately read the new state of affairs
        v1 = interface.read_sensor(intervention_var)
    
    except:
        return 10**12

    # TRANSFORM AND THEN EVOLVE

    # get the system back into its initial state and apply the transformation
    # (in a single step)
    interface.set_actuator(intervention_var, func.EvaluateAt([v0]))

    # evolve the system
    interface.set_actuator(time_var, time_interval)

    # read the final state of affairs
    v2 = interface.read_sensor(intervention_var)
    
    # check for overflow on sensors
    try:
        out = v1 - v2
    except:
        out = 10**6

    #DEBUGGING:
#    print "SymFunc output: function = {}, v1 = {}, v2 = {}, v1-v2 = {}".format(func._expression.Evaluate(),v1,v2,out)

    # COMPARE THE FINAL STATES
    return (out)

    

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
        check = FindParamVals(interface, func, time_var, intervention_var, time_interval, const_range)   
        if check == 1:
            #Test whether func with the generated constants is a symmetry
            sum += pow(SymFunc(interface, func, time_var, intervention_var, time_interval), 2)
        else:
            sum += 10**12
#            print func._expression.Evaluate() + ' could not find constants'
        
    return (sum/inductive_threshold)


def Loop(nextGeneration, current_generation, num_mutes, deck, seed_generation, interface, time_var, intervention_var, inductive_threshold, time_interval, const_range):
#    pdb.set_trace()
    for func in current_generation:
 
        # compute each function's error and combine with any previous data
        func._error = (func._error+(SymmetryGroup(interface, func, time_var, intervention_var, inductive_threshold, time_interval, const_range)))/2

        for i in range(num_mutes):
            #Combine the function with the functions in the deck in various ways
            modifiedFunc = randomTreeOperation(func, deck, seed_generation)
            #Measure fitness (convert errors to fitnesses)
            modifiedFunc._error = (SymmetryGroup(interface, modifiedFunc, time_var, intervention_var, inductive_threshold, time_interval, const_range))
            nextGeneration.append(modifiedFunc)
            interface.reset()
    return nextGeneration

def GeneticAlgorithm(cores, interfaces, seed_generation, time_var, intervention_var, inductive_threshold, time_interval, const_range, deck, generation_limit, num_mutes, generation_size, percent_guaranteed):
    """ Genetic Algorithm to find functions which are most likely to be symmetries
    
        Parameters:
            interface: List of cloned system interfaces each of which is used to modify/monitor the system
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
#    pdb.set_trace()
    current_generation = seed_generation
    # open output file for use in debugging
    f = open('out','w')

    for generation in range(0, generation_limit):
        nextGeneration = []
        #All members of the current generation are candidates for the next generation
        nextGeneration.extend(current_generation)
       
        #If this is the first generation, loop through and compute errors on the seed functions
        if generation == 0:
            for func in current_generation:
                func._error = (SymmetryGroup(interfaces[0], func, time_var, intervention_var, inductive_threshold, time_interval, const_range))
      
#        pdb.set_trace() 

        # Divide up the current generation into piles based on available cores and system clones
        num_piles= min(cores, len(interfaces))
        quot = divmod(len(current_generation),num_piles)
        job_data = []
        if len(current_generation) >= num_piles:
            for i in range(num_piles):
                job_data.append(current_generation[quot[0]*i:min((quot[0]*i+quot[0]), len(current_generation))])
        
            # create job server for parallel processing
            job_server = pp.Server(ncpus=num_piles)

            # create the jobs
            jobs = [job_server.submit(Loop,([], job_data[i], num_mutes, deck, seed_generation, interfaces[i], time_var, intervention_var, inductive_threshold, time_interval, const_range), (randomTreeOperation,SymmetryGroup,FindParamVals,SymFunc),("VABClasses","math","random","numpy","copy","pdb")) for i in range(num_piles)]   

            # acquire and process the output
            if (generation == 26):
                pdb.set_trace()
            out = []
            for i in range(len(jobs)):
                out.extend(jobs[i]())
            nextGeneration.extend(out)

            # destroy the job server
            job_server.destroy()

        else:
            #Loop through all of the current-gen functions
            Loop(nextGeneration, current_generation, num_mutes, deck, seed_generation, interfaces[0], time_var, intervention_var, inductive_threshold, time_interval, const_range)

#        pdb.set_trace()     
        #Sort the next generation by fitness 
        comparator = lambda x: x._error
        nextGeneration.sort(key=comparator)
       
        #Figure out how many are guaranteed to be passed on
        numGuaranteed = max(int(percent_guaranteed * len(nextGeneration)),1)

        #Check to see if we need to save more than the guaranteed percentage
        if generation_size > numGuaranteed:
            #Check whether there's room to save everything
            if generation_size > len(nextGeneration):
                current_generation = nextGeneration
                
                #DEBUGGING
#                print "Function with smallest error: {}. Error: {}".format(current_generation[0]._expression.Evaluate(), current_generation[0]._error)

                continue
               
            #We can't save everything.
            #First, preserve the most fit functions
            if numGuaranteed > 0:
                reducedGeneration = nextGeneration[0:numGuaranteed]
                nextGeneration = nextGeneration[numGuaranteed : len(nextGeneration)]
            
            #Create a list of running totals, representing a weighted distribution
            errorTotals = []
            runningTotal = 0
            for func in nextGeneration:
                runningTotal += func._error
                errorTotals.append(runningTotal)

            #Determine the number of slots to fill
            slots_remaining = generation_size - numGuaranteed

            #Fill the rest of the spots in the next generation
            for i in range(slots_remaining):
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
            current_generation = nextGeneration[0:(generation_size)]
        #DEBUGGING
        print "Generation: {}.\n".format(generation)
        f.write("Generation: {}.\n".format(generation))
        print "Lowest error function: {}. Mean Squared Error: {}.\n".format(current_generation[0].ExpressionString(), current_generation[0]._error)
        f.write("Lowest error function: {}. Mean Squared Error: {}.\n".format(current_generation[0].ExpressionString(), current_generation[0]._error))

    # close file for debug
    f.close()

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


def randomTreeOperation(function1, deck, seed):
    """Returns the result of a random combination of function trees 1 and 2
    """
    operations = [['*',2], ['+',2], ['-',2],['math.exp',1], ['/',2]]
    opCode = random.randint(0,len(operations)-1)
    replica = copy.deepcopy(function1)

    # First, decide whether to apply a constructive or destructive op
    if random.random() < 0.5:
        # choose a random member of the seed population
        term = random.randint(0,len(seed)-1)
        replica.ReplaceRandomNode(seed[term]._expression)
    elif operations[opCode][1] == 1:
        replica.OperateOnRandom(operations[opCode][0])
    else:
        # Draw a function at random from the deck
        function2 = deck[random.randint(0,len(deck)-1)]
        # apply the binary operation
        replica.OperateOnRandom(operations[opCode][0], function2)
    return replica
    

def RandomSelection(deck, num_selected):
    """Returns num_selected number of elements of the list, deck
    """
    random.shuffle(deck)
    return deck[0:(num_selected)]
