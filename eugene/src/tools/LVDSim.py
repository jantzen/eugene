#LVSim.py

""" A suite of tools for running LotkaVolterraSND simulations."""

from LotkaVolterraND import LotkaVolterraND
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def simpleSim(r, alpha, init_x, iterations, delta_t=1):
    """ Simulates a competitive Lotka-Volterra model with n 
            species
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            alpha -- the interaction matrix; a matrix of inter-species
                interaction terms, where a[i,j] is the effect of species j on
                the population of species i.
            init_x -- an array of species population size at the start of the
                observation period, where init_x[i] is the initial population
                of species i.
            iterations -- the number of times the system should be updated.
            delta_t -- the change to the time index each iteration.
                (default 1)
                
            Returns:
            x -- an array of species population size at the end of the
                observation period, where x[i] is the final population of 
                species i.
    """
    
    lv = LotkaVolterraND(r, alpha, init_x)
    #for t in trange(iterations):
    #    lv.update_x(delta_t)
    lv.update_x(iterations)
    return lv._x
    
    
def speciesAlive(populations, threshold=0.01):
    """ Returns the number of elements in array 'populations' that are larger
            than 'threshold'.
        
        Keyword arguments:
        populations -- an array of species populations
        threshold -- the size a population must be to be considered extant.
            (default 0.01)
            
        Returns:
        number -- the number of elements in array 'populations' that are larger
            than 'threshold'.
    """
    
    return sum(i > threshold for i in populations)
        
        
def simCompare(params1, params2, max_time, num_times, overlay):
    """ Compares two systems and returns two arrays of data that cover the same
            range.
            
            Keyword arguments:
            params1 -- an array of species growth rates "r", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item params1[0] shall be the first simulation's
                array of growth rates, and params1[1] shall be the first
                simulation's interaction matrix.
            params2 -- an array of species growth rates "r", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item params2[0] shall be the second simulation's
                array of growth rates, and params2[1] shall be the second
                simulation's interaction matrix.
            max_time -- the highest time value to sample the system at.
            num_times -- the number of times to sample the system between t=0
                and t=max_time.
            overlay -- a function that takes an array of data and returns an
                a new data array. This function is overlaid on the data.
                
            Returns:
            2d array -- two arrays of data from the systems that cover the same
                range.
    """
    
    lv1 = LotkaVolterraND(params1[0], params1[1], 0)
    lv2 = LotkaVolterraND(params2[0], params2[0], 0)
    times1 = np.random.uniform(0, max_time, num_times)
    times2 = np.random.uniform(0, max_time, num_times)
    xs1 = lv1.check_xs(times1)
    xs2 = lv2.check_xs(times2)
    f1 = overlay(xs1)
    f2 = overlay(xs2)
    data = (f1, f2)
    data = rangeCover(data)
    return data

def randInitPopsSim(r, alpha, iterations, delta_t=1):
    """ Simulates a competitive Lotka-Volterra model with n 
            species with initial populations selected from a uniform random 
            distribution.
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            alpha -- the interaction matrix; a matrix of inter-species
                interaction terms, where a[i,j] is the effect of species j on
                the population of species i.
            iterations -- the number of times the system should be updated.
            delta_t -- the change to the time index each iteration.
                (default 1)
                
            Returns:
            x -- an array of species population size at the end of the
                observation period, where x[i] is the final population of 
                species i.
    """
    
    init_x = np.random.uniform(size=np.size(r))
    
    return simpleSim(r, alpha, init_x, iterations, delta_t=1)

def runByArray(param_arr, iterations):
    """ Simulates many competitive Lotka-Volterra models of n 
            species with initial populations selected from a uniform random 
            distribution.
        
            Keyword arguments:
            param_arr -- an array of species growth rates "r", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item array[k][0] shall be the kth simulation's
                array of growth rates, and array[k][1] shall be the kth
                simulation's interaction matrix.
            iterations -- the number of times the system should be updated; i.e.
                the ammount of time to run each simulation.
                
            Returns:
            x_arr -- a list of arrays of species population sizes at the end of
                the observation period, where x[k][i] is the final population of 
                species i for the kth simulation.
    """
    
    #populations = []
    num_cores = multiprocessing.cpu_count() - 2
    results = Parallel(n_jobs=num_cores)(delayed(randInitPopsSim)(params[0], params[1], iterations) for params in param_arr)
    #for params in tqdm(param_arr):
    #    r = params[0]
    #    alpha = params[1]
    #    x = uniRandInitPopsSim(r, alpha, sigma, iterations)
    #    populations.append(x)
        
    #return populations
    return results
    

def resultsByPoint(param_arr, iterations, per_point):
    """ Simulates many competitive Lotka-Volterra models of n 
            species with initial populations selected from a uniform random 
            distribution. This will simulate each set of parameters per_point
            number of times, selecting random initial conditions for the system
            each time.
        
            Keyword arguments:
            param_arr -- an array of species growth rates "r", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item array[k][0] shall be the kth simulation's
                array of growth rates, and array[k][1] shall be the kth
                simulation's interaction matrix.
            iterations -- the number of times the system should be updated; i.e.
                the ammount of time to run each simulation.
            per_point -- the number of times the system should be simulated for
                each set of parameters
                
            Returns:
            x_arr -- a 3-D array, whose shape is (len(param_arr), per_point, n)
                x_arr[i] will give the final species populations for the ith
                set of parameters for all rounds. x_arr[i, j] will give 
                a list of arrays of species population sizes at the end of
                the observation period for the jth round of simulating the ith
                set of parameters, where x_arr[i, j, k] is the final population 
                of species k for the jth round of simulating the ith set of
                parameters.
    """
    
    species = np.size(param_arr[0][0])
    x_arr = np.ndarray((len(param_arr), per_point, species))
    for i in range(per_point):
        x_arr[:, i] = runByArray(param_arr, iterations)
        
    return x_arr
    

def rangeCover(data):
    """ Keyword arguments:
            data -- an array of 2-tuples of pairs of arrays, i.e. data[0] holds
                the records for f and f'. f and f' have the same length. E.g.
                array([[[ 0,  2,  8,  6,  4],
                    [99, 98, 97, 96, 95]],

                   [[ 1,  4,  2, 16, 32],
                    [89, 88, 87, 86, 85]]])
    """

    frames = []
    keys = []
    highestLow = min(data[0][0])
    lowestHigh = max(data[0][0])
    for tup in data:
        keys.append(tup[0])
        tupDf = pd.DataFrame(data=np.array(tup))
        frames.append(tupDf.sort_values([0, 1], 1))

        if min(tup[0]) > highestLow:
            highestLow = min(tup[0])
            
        if max(tup[0]) < lowestHigh:
            lowestHigh = max(tup[0])
            
    
    # then find intersection of arrays/dicts.
    #commonRange = reduce(np.intersect1d, (keys))
    
    # create tuples of selected range from original data
    output = []
    for frame in frames:
        mat = frame.as_matrix()
        keys = []
        values = []
        for i in range(len(mat[0])):
            if (mat[0][i] <= lowestHigh) and (mat[0][i] >= highestLow):
            #if mat[0][i] in commonRange:
                keys.append(mat[0][i])
                values.append(mat[1][i])
                
        output.append((keys, values))
        
    print(lowestHigh)
    print(highestLow)
    return output