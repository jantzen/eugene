# VABProcess.py

import time
import math
import random
import numpy as np
from VABClasses import *
import pdb
import pp
import scipy.stats as stats


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
        raise ValueError(
            'data list must have len >= 5 and contain an odd number of points.')
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

    # Put the system near the sensor limit (assume this doesn't break 
    # causal structure)
    max_x = interface.get_sensor_range(target_var)[1]
    interface.set_actuator(target_var,0.9*max_x)
    
    # Sample 5 points
    x = []
    try:
        for i in range(5):
            x.append(interface.read_sensor(target_var))
            time.wait(0.05)

        # Compute the derivative
        d2 = EmpiricalDeriv(x,0.05)
    except:
        # If the sensor threw an out-of-range error, the system must 
        # increase at that point, so that's the global max.
        global_max = interface.get_sensor_range(target_var)[1]
        


def FindParamVals(interface, func, time_var, intervention_var, 
                  time_interval, ROI):
    """ ROI: a dictionary of lists representing ranges
    """

    from scipy import optimize
    # get the range of possible values for target variable in the ROI
    [f_min,f_max] = ROI[intervention_var]

    # get the range of possible values for index variable from ROI
#    [t_min, t_max] = ROI[time_var]

    # set the tolerance
    tol = (float(f_max) - float(f_min)) / 10**6

    # choose a value in this range at random
    f_target = random.uniform(f_min,f_max)
   
    # read the current state of the system
    init_val = interface.read_sensor(intervention_var)
    
    # define functions that will allow a search for appropriate 
    # parameters
    f = lambda c,v: eval(func.ExpressionString())
    func_min = lambda c: (f(c, [init_val]) - f_target)**2

    # construct an initial guess (an array of 1s of length equal to the 
    # number of params)
    init_guess = np.array([1] * func.CountParameters())
    try: 
        sol = optimize.minimize(func_min, init_guess, tol=tol)
    except:
        return 0
    
    # if successful, set the parameters in func, reset the system, and 
    # return
    if sol.success == True and sol.fun < tol:
        func.SetParameters(sol.x.tolist())
        interface.set_actuator(intervention_var, random.uniform(f_min, f_max))
#        interface.set_actuator(time_var, random.uniform(t_min, t_max))
        return 1
    else:
        return 0



def SymFunc(interface, func, time_var, intervention_var, time_interval):

    """ interface: an interface object containing sensors and actuators
        func: the function (as a FunctionTree object) to be tested
        time_var: an integer indicating the ID of the index variable
        intervention_var: an integer giving the ID of the target
        variable
        time_interval: the interval over which to move the index
        variable
    """
    # EVOLVE AND THEN TRANSFORM
    
    # record the initial state 
    # (so it can be replicated in the next step)
    t0 = interface.read_sensor(time_var)
    v0 = interface.read_sensor(intervention_var)
    
    # EVOLVE the system
    interface.set_actuator(time_var, time_interval)

    # TRANSFORM the system

    # before attempting the transformation, make sure the function 
    # evaluates at the desired point (this is going to introduce more noise 
    # because the system is going to continue to evolve.
    
    try:
        set_point = func.EvaluateAt([interface.read_sensor(intervention_var)])
        interface.set_actuator(intervention_var, set_point)

    # immediately read the new state of affairs
        v1 = interface.read_sensor(intervention_var)
    
    except:
        return 10**12

    # TRANSFORM AND THEN EVOLVE

    # get the system back into its initial state and TRANSFORM 
    # (in a single step)
    interface.set_actuator(intervention_var, func.EvaluateAt([v0]))

    # evolve the system
    interface.set_actuator(time_var, time_interval)

    # read the final state of affairs
    v2 = interface.read_sensor(intervention_var)
    
    # COMPARE THE FINAL STATES (and check for overflow on sensors)
    try:
        out = v1 - v2
    except:
        out = 10**6
    # DEBUGGING
#    print "Function:{}, c[0] = {}, v0 = {}, v1 = {}, v2 = {}, out = {}".format(
#            func.ExpressionString(), func._parameters, v0, v1, v2, out)
    return (out)


def SymmetryGroup(interface, func, time_var, intervention_var, 
                  inductive_threshold, time_interval, ROI):
    """ Tests to see if several variations of func are symmetries.  
        Returns the mean of squares of the error
        This is the fitness function for the GA
        ROIs: A list containing the ranges in which each 
        constant in the function must reside trials: The number of different 
        random variations of constants to test
    """ 
    from scipy import integrate

    sum = 0
    for x in range(0, inductive_threshold):
        constants = [];
        check = FindParamVals(interface, func, time_var, 
                              intervention_var, time_interval, ROI)   
        if check == 1:
            # Randomly select an increment for the index variable that keeps 
            # it in the ROI
            [t_min, t_max] = ROI[time_var]
            # Reset the system (since it makes no sense to run time backward)
            interface.reset()
            shift = random.uniform(0.01*(t_max-t_min), 0.9*(t_max-t_min))
            interface.set_actuator(time_var,shift)

            # Test whether func with the generated paramter values 
            # is a symmetry (distinct from identity)
#            sum += pow(SymFunc(interface, func, time_var, 
#                       intervention_var, time_interval), 2)
#            pdb.set_trace()
            sum += pow(SymFunc(interface, func, time_var, intervention_var, time_interval), 2) 
        else:
            sum += 10**12

    # compute divergence from identity
    try:
        if check == 1:
            int_kernel = lambda x: (func.EvaluateAt([x])-x)**2
            [a,b] = ROI[intervention_var]
            divergence = integrate.quad(int_kernel, a, b)
        else:
            divergence = [10**12]
    except:
        divergence = [10**12]
    if divergence[0] < 0 or sum < 0:
        print 'Function: {}  divergence = {} a = {}  b = {}  Parameters = []'.format(func.ExpressionString(),divergence, a, b, func._parameters)
        raise ValueError
    return (sum/inductive_threshold + 1./(divergence[0]+1)) 
    # the '+1' in the denominator is to avoid division by zero


def Loop(nextGeneration, current_generation, num_mutes, deck, 
         seed_generation, interface, time_var, intervention_var, 
         inductive_threshold, time_interval, ROI):
    for func in current_generation:
 
        # compute each function's error and combine with any previous 
        # data
        func._error = (func._error+(SymmetryGroup(interface, func, 
                       time_var, intervention_var, inductive_threshold, 
                       time_interval, ROI)))/2

        for i in range(num_mutes):
            # Combine the function with the functions in the deck 
            # in various ways
            modifiedFunc = randomTreeOperation(func, deck, 
                                               seed_generation)

            #Measure fitness (convert errors to fitnesses)
            modifiedFunc._error = (SymmetryGroup(interface, 
                                   modifiedFunc, time_var, 
                                   intervention_var, 
                                   inductive_threshold, time_interval, 
                                   ROI))
            nextGeneration.append(modifiedFunc)
            interface.reset()
    return nextGeneration


def GeneticAlgorithm(cores, interfaces, seed_generation, time_var, 
                     intervention_var, inductive_threshold, time_interval, 
                     ROI, deck, generation_limit, num_mutes, generation_size, 
                     percent_guaranteed):
    """ Genetic Algorithm to find functions which are most likely to be 
        symmetries
    
        Parameters:
            interface: List of cloned system interfaces each of which is
                used to modify/monitor the system
            current_generation: The current generation of the GA
            time_var: The sensor ID of the time sensor
            intervention_var: The sensor ID of the actuator
            inductive_threshold: The number of trials before 
                generalizing
            time_interval: Wait time
            ROI: Ranges of values in region of interest for each variable 
            deck: Set of expressions used to mutate functions
                generation_limit: Maximum number of generations before 
                ending
            num_mutes: The number of mutations to apply to each member 
                of the current generation
            generation_size: The maximum allowable size of a single 
                generation
            percent_guaranteed: The top x% of fit functions are 
                guaranteed to be passed to the next generation
    """
    
    current_generation = seed_generation
    # open output file for use in debugging
    f = open('out','w')
    
    for generation in range(0, generation_limit):
      
        for func in current_generation:
            func._error = (SymmetryGroup(interfaces[0], func, 
                           time_var, intervention_var, 
                           inductive_threshold, time_interval, 
                           ROI))
 
        nextGeneration = []
        # All members of the current generation are candidates for the 
        # next generation
        nextGeneration.extend(current_generation)
      
        # Divide up the current generation into piles based on 
        # available cores and system clones
        num_piles= min(cores, len(interfaces))
        quot = divmod(len(current_generation),num_piles)
        job_data = []
        if len(current_generation) >= num_piles:
            for i in range(num_piles):
                job_data.append(
                    current_generation[quot[0]*i:min((quot[0]*i+quot[0]), 
                    len(current_generation))])
        
            # create job server for parallel processing
            job_server = pp.Server(ncpus=num_piles)

            # create the jobs
            jobs = [job_server.submit(Loop,([], job_data[i], num_mutes, 
                    deck, seed_generation, interfaces[i], time_var, 
                    intervention_var, inductive_threshold, time_interval, 
                    ROI), (randomTreeOperation,SymmetryGroup,
                    FindParamVals,SymFunc),("VABClasses","math","random",
                    "numpy","copy","pdb")) for i in range(num_piles)]   
#            pdb.set_trace()
            # acquire and process the output
            out = []
            for i in range(len(jobs)):
                out.extend(jobs[i]())
            nextGeneration.extend(out)

            # destroy the job server
            job_server.destroy()

        else:
            #Loop through all of the current-gen functions
            Loop(nextGeneration, current_generation, num_mutes, deck, 
                 seed_generation, interfaces[0], time_var, 
                 intervention_var, inductive_threshold, time_interval, 
                 ROI)

        #Sort the next generation by fitness 
        comparator = lambda x: x._error
        nextGeneration.sort(key=comparator)
       
        #Figure out how many are guaranteed to be passed on
        numGuaranteed = max(int(percent_guaranteed * 
                            len(nextGeneration)),1)

        # Check to see if we need to save more than the guaranteed 
        # percentage
        if generation_size > numGuaranteed:
            #Check whether there's room to save everything
            if generation_size > len(nextGeneration):
                current_generation = nextGeneration
                
                continue
               
            # We can't save everything.
            # First, preserve the most fit functions
            if numGuaranteed > 0:
                reducedGeneration = nextGeneration[0:numGuaranteed]
                nextGeneration = nextGeneration[
                                    numGuaranteed : len(nextGeneration)
                                    ]
            
            # Create a list of running totals, representing a weighted 
            # distribution
            errorTotals = []
            runningTotal = 0
            for func in nextGeneration:
                runningTotal += func._error
                errorTotals.append(runningTotal)

            #Determine the number of slots to fill
            slots_remaining = generation_size - numGuaranteed

            #Fill the rest of the spots in the next generation
            for i in range(slots_remaining):
                # Pull a random number from a uniform distribution 
                # on [0, 1]
                rand = random.random()

                # Assuming an exponential probability density function 
                # over error of the form exp(-x), find the corresponding 
                # error value (x)
                x = -math.log(rand)
                
                # Use the generated error value to select a function to 
                # preserve
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
        print "Lowest error function: {}. Mean Squared Error: {}.\n".format(
                current_generation[0].ExpressionString(), current_generation[0]._error)
        f.write("Lowest error function: {}. Mean Squared Error: {}.\n".format(
            current_generation[0].ExpressionString(), current_generation[0]._error))

    # close file for debug
    f.close()

    return current_generation
        
    
def BranchAndBound(interface, seed_func, time_var, intervention_var, 
                   inductive_threshold, time_interval, ROIs, 
                   deck, complexity_limit):
    """ Exaustive search of possible functions generated with 
        expressions from the deck
    
        Parameters:
            interface: System interface that is used to modify/monitor 
                the system
            seed_func: The initial function
            time_var: The sensor ID of the time sensor
            intervention_var: The sensor ID of the actuator
            inductive_threshold: The number of trials before 
                generalizing
            time_interval: Wait time
            ROIs: Ranges of possible values for the constants 
                in the functions
            deck: Set of expressions used to mutate functions
            complexity_limit: Maximum number of expressions before 
                ending
    """

    #All generated functions are stored in possibleSymmetries
    possibleSymmetries = [seed_func]
    for x in range(seed_func._const_count, complexity_limit):
        for func in possibleSymmetries:
            for func2 in deck:
                modifiedFuncs = allOperations(func, func2)
                for function in modifiedFuncs:
                    function._error = SymmetryGroup(interface, function,
                        time_var, intervention_var, inductive_threshold, 
                        time_interval, ROIs)
                possibleSymmetries.append(modifiedFuncs)
    return possibleSymmetries
    
      
def allOperations(function1, function2):
    """ Returns the result of each possible combination of functions 1 
        and 2
    """
    
    added = function1
    added.Add(function2)
    multiplied = function1
    multiplied.Multiply(function2)
    exponentiated = function1
    exponentiated.Power(function2)
    return [added, multiplied, exponentiated]


def randomOperation(function1, function2):
    """ Returns the result of one of the possible combinations of 
        functions 1 and 2 chosen at random (from a uniform 
        distribution)
    """

    operation = random.randint(0, 1)
    function1Copy = Function(function1._function, 
                             function1._const_count, function1._var_count)
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
    """ Returns the result of a random combination of function 
        trees 1 and 2
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


#---------------------Methods for statistical approach-------------------------#
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


def BuildSymModel(data_frame, index_var, target_var, sys_id, epsilon=0):
    # from the raw curves of target vs. index, build tranformation curves (e.g.,
    # target1 vs. target2)
    abscissa = data_frame._target_values[0]
    ordinates = data_frame._target_values[1:]
    
    # for each curve, fit a polynomial using 10-fold cross-validation to choose
    # the order
#    pdb.set_trace()
    polynomials = []
    sampled_data = []
    for curve in ordinates:
        # first try a linear fit
        order = 1
        
        # partition the data
        data = np.vstack((abscissa, curve))
        data = data.transpose()
        np.random.shuffle(data)
        partition = np.array_split(data, 10)

        # compute the error using each partition as validation set
        mse = []
        for p in range(len(partition)):
            # build training data
            training_set = np.empty([0,2],dtype='float64')
            for i in range(len(partition)):
                if i != p:
                    training_set = np.concatenate((training_set, partition[i]), 0)
            
            # fit polynomial
            x = training_set[:,0]
            y = training_set[:,1]
            fit = np.polyfit(x, y, 1)
            # compute error of fit against partitition p
            x = partition[p][:,0]
            y = partition[p][:,1]
            sum_square_error = 0
            for i in range(len(x)):
                sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
            mse.append(sum_square_error/len(x))

        # compute mean square error for first order
        mmse = np.mean(np.array(mse))
        best_fit_order = order

        # assess fits for higher-order polynomials until the minimum is passed
        loop = True

#        pdb.set_trace()
        while loop:
            order += 1
            # partition the data
            # data = np.vstack((times, curve))
            # data = data.transpose()
            np.random.shuffle(data)
            partition = np.array_split(data, 10)
    
            # compute the error using each partition as validation set
            mse = []
            for p in range(len(partition)):
               # build training data
               training_set = np.empty([0,2],dtype='float64')
               for i in range(len(partition)):
                   if i != p:
                       training_set = np.concatenate((training_set, partition[i]), 0)
               # fit polynomial
               x = training_set[:,0]
               y = training_set[:,1]
               fit = np.polyfit(x, y, order)
               # compute error of fit against partition p
               x = partition[p][:,0]
               y = partition[p][:,1]
               sum_square_error = 0
               for i in range(len(x)):
                   sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
               mse.append(sum_square_error/len(x))

            # compute mean square error for current order
            mmse_candidate = np.mean(np.array(mse))

            # if significantly better, keep it. If not, keep the old and halt.
            if (mmse - mmse_candidate) > epsilon:
                mmse = mmse_candidate
                best_fit_order = order
                best_fit = fit

            else:
                loop = False

            # cap the complexity
            if order >= 10:
                loop = False


        
        # using the best-fit order, fit the full data set
        x = data[:,0]
        y = data[:,1]
        best_fit = np.polyfit(x, y, best_fit_order)
        polynomials.append(best_fit)
        
#        # compute and save coeficient of determination
#        SStot = np.sum(np.power(curve - np.mean(y),2))
#        SSres = np.sum(np.power(np.polyval(best_fit, x) - y, 2))
#        R2 = 1. - SSres/SStot
#        R2vals.append(R2)
#   
#    # for the best-fit polynomials, use random subsets of the data to sample
#    # values of the coefficient of determination
#    R2_samples = [[] for i in range(len(polynomials))]
#    for i, poly in enumerate(polynomials):
#        y_complete = ordinates[i]
#        data_complete = np.vstack((abscissa, y_complete))
#        data_complete = data_complete.transpose()
#        for j in range(20):
#            np.random.shuffle(data_complete)
#            x = data_complete[1:int(len(data_complete)/2.),0]
#            y = data_complete[1:int(len(data_complete)/2.),1]
#            m = np.polyval(poly, x)
#            SStot = np.sum(np.power(y - np.mean(y),2))
#            SSres = np.sum(np.power(m - y, 2))
#            R2 = 1. - SSres/SStot
#            R2_samples[i].append(R2)
        sampled_data.append(data)
    
    # build and output a SymModel object
    return SymModel(index_var, target_var, sys_id, sampled_data, polynomials, epsilon)


def SymTestTemporal(model, interface, time_var, target_var, ROI, num_trans,
        resolution=[100,1], alpha=0.05):

    # make a copy of the polynomial (symmetry transformation) list
    poly = copy.deepcopy(model._polynomials)

    # choose num_trans symmetry transformations at random from the model
#    if num_trans >  len(model._polynomials):
#       num_trans = len(model._polynomials)
#
#    if num_trans < len(model._polynomials):
#        np.random.shuffle(poly)
#        poly = poly[0:num_trans]
    
    # figure the delay to set between samples
    [time_low, time_high] = ROI[time_var]
    delay = (float(time_high) - float(time_low)) / resolution[0]

    # fill out the index variable (time) values 
    times = np.arange(time_low, time_high+delay, delay)
   
    success = True

    # for each transformation, test to see whether it works on sys
    for counter, trans in enumerate(poly):
        # sample data from system 
        samples = []
        
        # reset the system, then wait until time_low
        interface.reset()
        
        x0_tentative = interface.read_sensor(target_var)
        if np.polyval(trans, x0_tentative) > ROI[target_var][1]:
            x0 = ROI[target_var][0]
        elif np.polyval(trans, x0_tentative) < ROI[target_var][0]:
            x0 = ROI[target_var][1]
        else:
            x0 = x0_tentative

        interface.set_actuator(target_var, x0)
        time.sleep(time_low)
    
        # initialize the output list
        initial_data = []
    
        # start sampling
        for i in range(len(times)):
            initial_data.append(interface.read_sensor(target_var))
            time.sleep(delay)
    
        # convert the data 
        initial_data = np.array(initial_data)

        # now, transform the initial value using trans, and sample again
        x0_trans = np.polyval(trans, x0)
 
        # initialize the output list
        transformed_data = []
        
        interface.reset()
        interface.set_actuator(target_var, x0_trans)
        time.sleep(time_low)

        # start sampling 
        for i in range(len(times)):
            transformed_data.append(interface.read_sensor(target_var))
            time.sleep(delay)
    
        # convert the data 
#        transformed_data = np.array(transformed_data)
        
        # compute the expected curve
#        expected_data = np.polyval(trans, initial_data)

        # determine the order of the model
        order = len(trans) - 1

        # compute a sample of R^2 values
#        SStot = np.sum(np.power(transformed_data -
#            np.mean(transformed_data),2))
#        SSres = np.sum(np.power((expected_data - transformed_data),2))
#        R2 = 1. - SSres/SStot

        # if no good, stop here and return failure
#        if R2 < model._R2 - R2diff:
#            success = False
#            return success
        pdb.set_trace()
        # for the best-fit polynomials, use random subsets of the data to sample
        # values of the coefficient of determination
        R2_samples = []
        data_complete = np.vstack((initial_data, transformed_data))
        data_complete = data_complete.transpose()
        for j in range(20):
            np.random.shuffle(data_complete)
            x = data_complete[1:int(len(data_complete)/2.),0]
            y = data_complete[1:int(len(data_complete)/2.),1]
            m = np.polyval(trans, x)
            SStot = np.sum(np.power(y - np.mean(y),2))
            SSres = np.sum(np.power(m - y, 2))
            R2 = 1. - SSres/SStot
            R2_samples.append(R2)
        
        # compare the R2 sample with the corresponding sample from the model
        [D, p_val] = stats.ks_2samp(R2_samples, model._R2_samples[counter])

        if p_val < alpha:
            return False
        else:
            return True


def LackOfFitTest(sampled_data, expected_data, order, alpha):
    # compute the mean square for pure error
    Ntotal = 5 * len(sampled_data)
    Nunique = len(sampled_data)

    total = 0
    for i in range(Nunique):
        for j in range(5):
            total += np.power(sampled_data[i][j] - np.mean(sampled_data[i]), 2.)
    
    sigma_r2 = 1./(Ntotal - Nunique) * total

    # compute the mean square for lack of fit
    total = 0
    for i in range(Nunique):
        total += 5. * np.power(np.mean(sampled_data[i]) - expected_data[i], 2.)
    
    sigma_m2 = 1./(Nunique - order) * total

    pdb.set_trace()

    # compute F-statistic
    F = sigma_m2 / sigma_r2

    # compute p-value
    p = 1. - stats.f.cdf(F, Nunique - order, Ntotal - Nunique)

    if p < alpha:
        return True # the model doesn't fit
    else:
        return False # the model does fit


def CompareModels(model1, model2):
    """ Tests whether the models (and the systems they model) are equivalent. If
    so, it returns a combined model.
    """
#    pdb.set_trace()
    p_vals = []

    # initialize containers for data that may be passed out
    combined_sampled_data = []
    combined_polynomials = []

    for counter, poly1 in enumerate(model1._polynomials):
        # import relevant data
        poly2 = model2._polynomials[counter]
        data1 = model1._sampled_data[counter]
        data2 = model2._sampled_data[counter]

        # fit the joint data
        data = np.vstack((data1,data2))
        combined_sampled_data.append(data)

        # first try a linear fit
        order = 1
        
        # partition the data
        np.random.shuffle(data)
        partition = np.array_split(data, 10)

        # compute the error using each partition as validation set
        mse = []
        for p in range(len(partition)):
            # build training data
            training_set = np.empty([0,2],dtype='float64')
            for i in range(len(partition)):
                if i != p:
                    training_set = np.concatenate((training_set, partition[i]), 0)
            
            # fit polynomial
            x = training_set[:,0]
            y = training_set[:,1]
            fit = np.polyfit(x, y, 1)
            # compute error of fit against partitition p
            x = partition[p][:,0]
            y = partition[p][:,1]
            sum_square_error = 0
            for i in range(len(x)):
                sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
            mse.append(sum_square_error/len(x))

        # compute mean square error for first order
        mmse = np.mean(np.array(mse))
        best_fit_order = order

        # assess fits for higher-order polynomials until the minimum is passed
        loop = True       
        while loop:
            order += 1
            # partition the data
            # data = np.vstack((times, curve))
            # data = data.transpose()
            np.random.shuffle(data)
            partition = np.array_split(data, 10)
    
            # compute the error using each partition as validation set
            mse = []
            for p in range(len(partition)):
               # build training data
               training_set = np.empty([0,2],dtype='float64')
               for i in range(len(partition)):
                   if i != p:
                       training_set = np.concatenate((training_set, partition[i]), 0)
               # fit polynomial
               x = training_set[:,0]
               y = training_set[:,1]
               fit = np.polyfit(x, y, order)
               # compute error of fit against partition p
               x = partition[p][:,0]
               y = partition[p][:,1]
               sum_square_error = 0
               for i in range(len(x)):
                   sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
               mse.append(sum_square_error/len(x))

            # compute mean square error for current order
            mmse_candidate = np.mean(np.array(mse))

            # if significantly better, keep it. If not, keep the old and halt.
            if (mmse - mmse_candidate) > min(model1._epsilon, model2._epsilon):
                mmse = mmse_candidate
                best_fit_order = order
                best_fit = fit

            else:
                loop = False
            
#            # cap the complexity
#            if order >= 10:
#                loop = False

        # using the best-fit order, fit the full data set
        x = data[:,0]
        y = data[:,1]
        null_hyp = np.polyfit(x, y, best_fit_order)

        # save the best fit polynomial
        combined_polynomials.append(null_hyp)

        # now use the same 'model' (the same order polynomial) to fit each data
        # set individually
        poly1 = np.polyfit(data1[:,0], data1[:,1], best_fit_order)
        poly2 = np.polyfit(data2[:,0], data2[:,1], best_fit_order)

        # compute sum of squares (SS) and degrees of freedom (df) for the null
        # hypothesis: both data sets are described by the same polynomial

        # first, construct a function to norm the data
        norm = lambda x: (x - np.mean(data[:,1])) / (np.max(data[:,1]) -
            np.min(data[:,1])) 

        # SS null
        SSnull = 0
        for i in range(len(data)):
            x = data[i,0]
            y = data[i,1]
            SSnull += np.power(norm(y) - norm(np.polyval(null_hyp, x)), 2.)

        # df null
        df_null = len(data) - len(null_hyp)

        # SS alt
        SSalt = 0
        for i in range(len(data1)):
            x = data1[i, 0]
            y = data1[i, 1]
            SSalt += np.power(norm(y) - norm(np.polyval(poly1, x)), 2.)

        for i in range(len(data2)):
            x = data2[i, 0]
            y = data2[i, 1]
            SSalt += np.power(norm(y) - norm(np.polyval(poly2, x)), 2.)
        
        # df alt
        df_alt = len(data1) - len(poly1) + len(data2) - len(poly2)

        # F-statistic
        F = ((SSnull - SSalt)/SSalt)/((float(df_null) -
            float(df_alt))/float(df_alt))

        # p-value
        p = 1. - stats.f.cdf(F, (df_null - df_alt), df_alt) 

        p_vals.append(p)

#    pdb.set_trace()

    # if most of the p_vals exceed alpha = 0.05, then conclude that the models
    # are equivalent and return the new combined model; otherwise, return an
    # empty list. 
    p_vals = np.array(p_vals)
    if (np.sum(np.greater(p_vals, np.ones(p_vals.shape)*0.05)) >
         round(len(p_vals)/2.)):
        return SymModel(model1._index_var, model1._target_var, model1._sys_id,
                combined_sampled_data, combined_polynomials,
                min(model1._epsilon, model2._epsilon))
    else:
        return None


def Classify(system_ids, models):
    """ Assumes that the ith model corresponds to sys_id i.
    """
#    pdb.set_trace()
    # initialize the sort with the first system in the list of systems
    classes = []
    classes.append(Category(set([system_ids[0]]), models[0]))

    # sort the remainder of the systems
    for sys_id in system_ids[1:]:
        categorized = False
        for c in classes:
            # compare the unknown system to the paradigm
            result = CompareModels(models[sys_id], c._paradigm)
            if result != None:
                categorized = True
                c.add_system(sys_id)
                c.update_paradigm(result)
                break
        # if the system doesn't fit a known category, make a new one
        if categorized == False:
            classes.append(Category(set([sys_id]), models[sys_id]))

    # return the list of classes
    return classes
