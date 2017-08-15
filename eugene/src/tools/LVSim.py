#LVSim.py

""" A suite of tools for running LotkaVolterraSND simulations."""

from LotkaVolterraSND import LotkaVolterraSND
from tqdm import tqdm, trange
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def simpleSim(r, k, alpha, sigma, init_x, iterations, delta_t=1):
    """ Simulates a stochastic competitive Lotka-Volterra model with n 
            species
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            k -- an array of species carrying capacities, where k[i] is the 
               carrying capacity of species i. 
            alpha -- the interaction matrix; a matrix of inter-species
                interaction terms, where a[i,j] is the effect of species j on
                the population of species i.
            sigma -- an array of noise intensities where s[i] is the intensity
                of noise affecting species i.
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
    
    lv = LotkaVolterraSND(r, k, alpha, sigma, init_x)
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
    
    
def sameNoiseSim(r, k, alpha, sigma, init_x, iterations, delta_t=1):
    """ Simulates a stochastic competitive Lotka-Volterra model with n 
            species, and the same intrinsic noise for all species.
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            k -- an array of species carrying capacities, where k[i] is the 
               carrying capacity of species i.
            alpha -- the interaction matrix; a matrix of inter-species
                interaction terms, where a[i,j] is the effect of species j on
                the population of species i.
            sigma -- a noise intensity affecting all species.
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
    
    sigma_arr = np.full(len(r), sigma)
    return simpleSim(r, k, alpha, sigma_arr, init_x, iterations, delta_t)
    

def uniRandInitPopsSim(r, k, alpha, sigma, iterations, delta_t=1):
    """ Simulates a stochastic competitive Lotka-Volterra model with n 
            species with initial populations selected from a uniform random 
            distribution.
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            k -- an array of species carrying capacities, where k[i] is the 
               carrying capacity of species i. 
            alpha -- the interaction matrix; a matrix of inter-species
                interaction terms, where a[i,j] is the effect of species j on
                the population of species i.
            sigma -- a noise intensity affecting species. If the size of sigma
                is 1, then the same ammount of noise affects all species. 
                Otherwise it is assumed that sigma is an array of noise 
                intensities where s[i] is the intensity of noise affecting 
                species i.
            iterations -- the number of times the system should be updated.
            delta_t -- the change to the time index each iteration.
                (default 1)
                
            Returns:
            x -- an array of species population size at the end of the
                observation period, where x[i] is the final population of 
                species i.
    """
    
    init_x = np.random.uniform(size=np.size(r))
    
    if np.size(sigma) is 1:
        return sameNoiseSim(r, k, alpha, sigma, init_x, iterations, delta_t=1)
    elif np.size(sigma) is np.size(r):
        return simpleSim(r, k, alpha, sigma, init_x, iterations, delta_t=1)
    else:
        raise ValueError('Inorrect variable specification. Size of sigma must be 1 or the number of species')


def runByArray(param_arr, sigma, iterations):
    """ Simulates many stochastic competitive Lotka-Volterra models of n 
            species with initial populations selected from a uniform random 
            distribution.
        
            Keyword arguments:
            param_arr -- an array of species growth rates "r", an array of 
                species carrying capacities "k", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                k[i] is the carrying capacity of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item array[k][0] shall be the kth simulation's
                array of growth rates, array[k][1] shall be the kth
                simulation's carrying capacity, and array[k][2] shall be the
                kth simulation's interaction matrix.
            sigma -- a noise intensity affecting species. If the size of sigma
                is 1, then the same ammount of noise affects all species. 
                Otherwise it is assumed that sigma is an array of noise 
                intensities where s[i] is the intensity of noise affecting 
                species i.
            iterations -- the number of times the system should be updated; i.e.
                the ammount of time to run each simulation.
                
            Returns:
            x_arr -- a list of arrays of species population sizes at the end of
                the observation period, where x[k][i] is the final population of 
                species i for the kth simulation.
    """
    
    #populations = []
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(uniRandInitPopsSim)(params[0], params[1], params[2], sigma, iterations) for params in tqdm(param_arr))
    #for params in tqdm(param_arr):
    #    r = params[0]
    #    alpha = params[1]
    #    x = uniRandInitPopsSim(r, alpha, sigma, iterations)
    #    populations.append(x)
        
    #return populations
    return results
    

def resultsByPoint(param_arr, sigma, iterations, per_point):
    """ Simulates many stochastic competitive Lotka-Volterra models of n 
            species with initial populations selected from a uniform random 
            distribution. This will simulate each set of parameters per_point
            number of times, selecting random initial conditions for the system
            each time.
        
            Keyword arguments:
            param_arr -- an array of species growth rates "r", an array of 
                species carrying capacities "k", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                k[i] is the carrying capacity of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item array[k][0] shall be the kth simulation's
                array of growth rates, array[k][1] shall be the kth
                simulation's carrying capacity, and array[k][2] shall be the
                kth simulation's interaction matrix.
            sigma -- a noise intensity affecting species. If the size of sigma
                is 1, then the same ammount of noise affects all species. 
                Otherwise it is assumed that sigma is an array of noise 
                intensities where s[i] is the intensity of noise affecting 
                species i.
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
    for i in trange(per_point):
        x_arr[:, i] = runByArray(param_arr, sigma, iterations)
        
    return x_arr