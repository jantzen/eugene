# LVDSim.py

""" A suite of tools for running LotkaVolterraSND simulations."""

# from LotkaVolterraND import LotkaVolterraND
from eugene.src.virtual_sys.LotkaVolterraND import LotkaVolterraND
from eugene.src.auxiliary.probability import *
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from tqdm import tqdm, trange
import pdb


# Classes
##############################################################################
class Conditional_Density(object):
    def __init__(self, kde, x_range=[-np.inf, np.inf]):
        self._kde = kde
        self._xrange = x_range


    def density(self, y, x):
        """ Computes 1-D conditional distribution P(y | X=x). X is presumed
                to refer to the untransformed value, and y to the transformed value
                of a target variable.
            """

        if type(y) == np.ndarray:
            y = y.reshape((len(y), 1))
        elif type(y) == list:
            y = np.array(y)
            y = y.reshape((len(y), 1))
        else:
            y = np.array([y])
            y = y.reshape((len(y), 1))
        x = x * np.ones(y.shape)
        sample = np.hstack((x, y))

        p_xy = np.exp(self._kde.score_samples(sample))

        # compute unconditional probability of X = x
        func = lambda z: np.exp(self._kde.score_samples(np.array([x,
                                                                  z]).reshape(1, -1)))
        temp = quad(func, self._xrange[0],
                    self._xrange[1], epsabs=1. * 10 ** (-6), limit=30)
        p_x = temp[0]

        if not np.isfinite(p_x):
            raise ValueError("p_x did not evaluate to a finite number")

        if not np.isfinite(p_xy):
            raise ValueError("p_xy did not evaluate to a finite number")

        return (p_xy / p_x)


# Functions
##############################################################################


def simpleSim(r, k, alpha, init_x, iterations, delta_t=1):
    """ Simulates a competitive Lotka-Volterra model with n 
            species
        
            Keyword arguments:
            r -- an array of species growth rates, where r[i] is the growth
                rate of species i.
            k -- an array of species carrying capacities, where k[i] is the 
               carrying capacity of species i. 
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

    lv = LotkaVolterraND(r, k, alpha, init_x)
    # for t in trange(iterations):
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


def simData(params, max_time, num_times, overlay):
    """ Generates data for a list of parameters corresponding to systems and 
    returns a list of arrays of data that cover the same range.
            
            Keyword arguments:
            params1 -- an array of species growth rates "r", an array of 
                species carrying capacities "k", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                k[i] is the carrying capacity of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item params1[0] shall be the first simulation's
                array of growth rates, params1[1] shall be the first
                simulation's carrying capacity, params1[2] shall be the first
                simulation's interaction matrix, and params1[3] shall be the
                first simulation's initial populations.
            params2 -- an array of species growth rates "r", an array of 
                species carrying capacities "k", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                k[i] is the carrying capacity of species i,
                and alpha[i,j] is the effect of species j on the population of 
                species i. Item params2[0] shall be the second simulation's
                array of growth rates, params2[1] shall be the second
                simulation's carrying capacity, params2[2] shall be the second
                simulation's interaction matrix, and params2[3] shall be the
                first populations's initial popoulations.
            max_time -- the highest time value to sample the system at.
            num_times -- the number of times to sample the system between t=0
                and t=max_time.
            overlay -- a function that takes an array of data and returns an
                a new data array. This function is overlaid on the data.
                
            Returns:
            2d array -- two arrays of data from the systems that cover the same
                range.
    """

    lv = []
    lv_trans = []
    for param_set in params:
        lv.append(LotkaVolterraND(param_set[0], param_set[1], param_set[2], param_set[3], 0))
        lv_trans.append(LotkaVolterraND(param_set[0], param_set[1], param_set[2], param_set[3], 0))

    times = []
    times_trans = []
    for i in range(len(lv)):
        times.append(np.sort(np.random.uniform(0, max_time, num_times)))
    for i in range(len(lv_trans)):
        times_trans.append(np.sort(np.random.uniform(0, max_time, num_times)))

    xs = []
    xs_trans = []
    for i, sys in enumerate(lv):
        xs.append(sys.check_xs(times[i]))
    for i, sys in enumerate(lv_trans):
        xs_trans.append(sys.check_xs(times[i]))

    raw_data = []
    for i in range(len(lv)):
        f = overlay(xs[i])
        f_trans = overlay(xs_trans[i])
        raw_data.append([f, f_trans])

    raw_data = np.array(raw_data)

    data = rangeCover(raw_data)

    return data


def randInitPopsSim(r, k, alpha, iterations, delta_t=1):
    """ Simulates a competitive Lotka-Volterra model with n 
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
            iterations -- the number of times the system should be updated.
            delta_t -- the change to the time index each iteration.
                (default 1)
                
            Returns:
            x -- an array of species population size at the end of the
                observation period, where x[i] is the final population of 
                species i.
    """

    init_x = np.random.uniform(size=np.size(r))

    return simpleSim(r, k, alpha, init_x, iterations, delta_t=1)


def runByArray(param_arr, iterations):
    """ Simulates many competitive Lotka-Volterra models of n 
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
            iterations -- the number of times the system should be updated; i.e.
                the ammount of time to run each simulation.
                
            Returns:
            x_arr -- a list of arrays of species population sizes at the end of
                the observation period, where x[k][i] is the final population of 
                species i for the kth simulation.
    """

    # populations = []
    num_cores = multiprocessing.cpu_count() - 2
    results = Parallel(n_jobs=num_cores)(
        delayed(randInitPopsSim)(params[0], params[1], params[2], iterations) for params in param_arr)
    # for params in tqdm(param_arr):
    #    r = params[0]
    #    alpha = params[1]
    #    x = uniRandInitPopsSim(r, alpha, sigma, iterations)
    #    populations.append(x)

    # return populations
    return results


def resultsByPoint(param_arr, iterations, per_point):
    """ Simulates many competitive Lotka-Volterra models of n 
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
    # commonRange = reduce(np.intersect1d, (keys))

    # create tuples of selected range from original data
    output = []
    for frame in frames:
        mat = frame.as_matrix()
        keys = []
        values = []
        for i in range(len(mat[0])):
            if (mat[0][i] <= lowestHigh) and (mat[0][i] >= highestLow):
                # if mat[0][i] in commonRange:
                keys.append(mat[0][i])
                values.append(mat[1][i])

        output.append((keys, values))

    print(lowestHigh)
    print(highestLow)
    return output


def downSample(data):
    """ data is presumed to have the form of the output of simCompare and
    rangeCover.
    """

    # find the length of the shortest tuple
    lengths = []
    for tup in data:
        lengths.append(len(tup[0]))

    cut = min(lengths)

    cut_data = []

    for tup in data:
        cut_tup = []
        for segment in tup:
            cut_segment = np.random.choice(segment, cut)
            cut_tup.append(cut_segment)
        cut_data.append(cut_tup)

    return cut_data


def tuplesToBlocks(data):
    """ Converts a list of 2-member lists of data to a list of 2-d numpy arrays
    of data.
    """

    out = []
    for tup in data:
        col1 = np.array(tup[0])[:, np.newaxis]
        col2 = np.array(tup[1])[:, np.newaxis]
        out.append(np.hstack((col1, col2)))

    return out


def blocksToKDEs(data):
    """ For a list of 2-D arrays of data, uses kernel density esitmation to
    estimate joint probability densities, and outpus a list of trained sklearn KernelDensity
    objects.
    """
    kde_objects = []
    for block in data:
        kde = KernelDensity(bandwidth=0.5)
        kde.fit(block)
        kde_objects.append(kde)

    return kde_objects


def KDEsToDensities(kde_objects):
    """ Converts a list of trained sklearn KernelDensity objects to a list of
    two-argument functions (joint probability densities).
    """
    densities = []
    for kde in kde_objects:
        func = lambda x, y, kde=kde: np.exp(kde.score_samples(np.array([x,
                                                                        y]).reshape(1, -1)))
        # note the dummy variable used above to capture the current kde value
        densities.append(func)

    return densities


def jointToConditional(joint_densities, x_range=[-np.inf, np.inf]):
    out = []
    for joint in joint_densities:
        c = Conditional_Density(joint, x_range)
        out.append(c.density)

    return out


def blocksToScipyDensities(data):
    """ For a list of 2-D arrays of data, uses kernel density esitmation to
    estimate joint probability densities, and outpus a list of trained sklearn KernelDensity
    objects.
    """
    densities = []
    for block in data:
        kde = stats.gaussian_kde(block)
        pdf = kde.evaluate(block)
        densities.append(pdf)

    return densities


def meanHellinger(func1, func2, x_range):
    def integrand(x):
        f1 = lambda y: func1(y, x)
        f2 = lambda y: func2(y, x)

        return HellingerDistance(f1, f2, x_range)


    out = quad(integrand, x_range[0], x_range[1], epsabs=1. * 10 ** (-6), limit=30)

    return out[0] / (float(x_range[1]) -
                     float(x_range[0]))


def distanceH(densities, x_range=[-np.inf, np.inf]):
    """ Returns a distance matrx.
    """
    s = len(densities)
    dmat = np.zeros((s, s))

    for i in trange(s):
        for j in trange(i + 1, s):
            # func_i = lambda y: densities[i](y, 0.5)
            # func_j = lambda y: densities[j](y, 0.5)
            # dmat[i,j] = HellingerDistance(func_i, func_j, x_range)
            # dmat[j,i] = dmat[i,j]
            dmat[i, j] = meanHellinger(densities[i], densities[j], x_range)
            dmat[j, i] = dmat[i, j]

    return dmat


def distanceH2D(densities, x_range=[-np.inf, np.inf]):
    """ Returns a distance matrx.
    """
    s = len(densities)
    dmat = np.zeros((s, s))

    for i in trange(s):
        for j in trange(i + 1, s):
            dmat[i, j] = Hellinger2D(densities[i], densities[j], x_range[0],
                                     x_range[1])
            dmat[j, i] = dmat[i, j]

    return dmat


def meanEuclidean(func1, func2, x_range):
    def integrand(x):
        f1 = lambda y: func1(y, x)
        f2 = lambda y: func2(y, x)

        return EuclideanDistance(f1, f2, x_range)


    out = quad(integrand, x_range[0], x_range[1])

    return out[0] / (float(x_range[1]) -
                     float(x_range[0]))


def distanceL2(densities, x_range=[-10, 10]):
    """ Returns a distance matrx.
    """
    s = len(densities)
    dmat = np.zeros((s, s))

    for i in range(s):
        for j in range(i + 1, s):
            # func_i = lambda y: densities[i](y, 0.5)
            # func_j = lambda y: densities[j](y, 0.5)
            # dmat[i,j] = EuclideanDistance(func_i, func_j, x_range)
            # dmat[j,i] = dmat[i,j]
            dmat[i, j] = meanEuclidean(densities[i], densities[j], x_range)
            dmat[j, i] = dmat[i, j]

    return dmat
