#LVSim.py

""" A suite of tools for running LotkaVolterraSND simulations."""

from LotkaVolterraND import LotkaVolterraND
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def simpleSim(r, alpha, init_x, iterations, delta_t=1):
    """ Simulates a stochastic competitive Lotka-Volterra model with n 
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
        

def randInitPopsSim(r, alpha, iterations, delta_t=1):
    """ Simulates a stochastic competitive Lotka-Volterra model with n 
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
    """ Simulates many stochastic competitive Lotka-Volterra models of n 
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
    """ Simulates many stochastic competitive Lotka-Volterra models of n 
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