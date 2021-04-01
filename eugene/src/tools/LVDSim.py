# LVDSim.py

""" A suite of tools for running LotkaVolterraSND simulations."""

# from LotkaVolterraND import LotkaVolterraND
from eugene.src.virtual_sys.LotkaVolterraND import LotkaVolterraND
from eugene.src.virtual_sys.LotkaVolterraSND import LotkaVolterraSND
from eugene.src.virtual_sys.LotkaVolterra2OND import LotkaVolterra2OND
from eugene.src.virtual_sys.LotkaVolterraNDLin import LotkaVolterraNDLin
from eugene.src.auxiliary.probability import *
import random
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.integrate import quad
from scipy import stats
from tqdm import tqdm, trange
import eugene.src.auxiliary.sampling.resample as resample
from multiprocessing import cpu_count
import copy


# import pdb


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
                                                                  z]).reshape(1,
                                                                              -1)))
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


def simData(params, max_time, num_times, overlay, stochastic_reps=None,
        range_cover=True):
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
    if stochastic_reps is None:
        for param_set in params:
            lv.append(LotkaVolterraND(param_set[0], param_set[1], param_set[2], param_set[3], 0))
            lv_trans.append(LotkaVolterraND(param_set[0], param_set[1], 
                param_set[2], param_set[4], 0))
    else:
        for param_set in params:
            lv.append(LotkaVolterraSND(param_set[0], param_set[1], param_set[2],
                param_set[3], param_set[4], 0))
            lv_trans.append(LotkaVolterraSND(param_set[0], param_set[1], 
                param_set[2], param_set[3], param_set[5], 0))
 
    times = []
    times_trans = []
    for i in range(len(lv)):
        times.append(np.linspace(0., max_time, num_times))
    for i in range(len(lv_trans)):
        times_trans.append(np.linspace(0., max_time, num_times))

    if stochastic_reps is None:
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

    else:
        xs = []
        xs_trans = []
        for i, sys in enumerate(lv):
            reps = []
            init_x = copy.deepcopy(sys._init_x)
            # temp = sys.check_xs(times[i])
            # sys._x = init_x
            for r in range(stochastic_reps):
                # reps.append(sys.check_xs(times[i]))
                reps.append(sys.check_xs(times[i]).T)
                # temp = np.vstack((temp,sys.check_xs(times[i])))
                sys._x = copy.copy(init_x)
            # xs.append(temp)
            xs.append(reps)

        for i, sys in enumerate(lv_trans):
            reps_trans = []
            init_x = copy.deepcopy(sys._init_x)
            # temp = sys.check_xs(times[i])
            # sys._x = init_x
            for r in range(stochastic_reps):
                # reps_trans.append(sys.check_xs(times[i]))
                reps_trans.append(sys.check_xs(times[i]).T)
                # temp = np.vstack((temp,sys.check_xs(times[i])))
                sys._x = copy.copy(init_x)
            # xs_trans.append(temp)
            xs_trans.append(reps_trans)

        raw_data = []
        for i in range(len(lv)):
            f = overlay(xs[i])
            f_trans = overlay(xs_trans[i])
            raw_data.append([f, f_trans])

    if range_cover:
        data, high, low = rangeCover(raw_data)
        return data, low, high

    else:
        return raw_data


def simDataLin(params, max_time, num_times, overlay, range_cover=False):
    """ Generates data for a list of parameters corresponding to systems and
    returns a list of arrays of data that cover the same range.

            Keyword arguments:
            params[n] -- an array of species growth rates "r", an array of
                species carrying capacities "k", and interaction
                matrices "alpha"; where r[i] is the growth rate of species i,
                k[i] is the carrying capacity of species i,
                and alpha[i,j] is the effect of species j on the population of
                species i. Item params1[0] shall be the first simulation's
                array of growth rates, params1[1] shall be the first
                simulation's carrying capacity, params1[2] shall be the first
                simulation's interaction matrix, and params1[3] shall be the
                first simulation's initial populations, params1[4] shall be the
                first simulation's transformed initial populations, and
                params1[5] shall be the scale of the non-linear term in the
                equation.
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
        lv.append(LotkaVolterraNDLin(param_set[0], param_set[1], param_set[2],
                                     param_set[3], param_set[5], 0))
        lv_trans.append(LotkaVolterraNDLin(param_set[0], param_set[1],
                                           param_set[2], param_set[4],
                                           param_set[5], 0))
    times = []
    times_trans = []
    for i in range(len(lv)):
        times.append(np.linspace(0., max_time, num_times))
    for i in range(len(lv_trans)):
        times_trans.append(np.linspace(0., max_time, num_times))

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

    if range_cover:
        data, high, low = rangeCover(raw_data)
        return data, low, high

    else:
        return raw_data


def simData2OD(params, max_time, num_times, overlay, range_cover=False):
    """ Generates data for a list of parameters corresponding to systems and
    returns a list of arrays of data that cover the same range.

            Keyword arguments:
            prams -- a list of parameters for each system desired to simulate.
                params[n][0] is an array-like of species growth rates "r" for
                species in system n. params[n][1] is an array-like of species
                carrying capacities "k". params[n][2] is the interaction matrix
                "alpha". params[n][3] is an array-like of initial populations.
                params[n][4] is an array-like of initial populations for the
                transformed version of the system. params[n][5] is an array-like
                of species ~growth velocities~. params[n][6] is a scalar that
                dictates how strong the second order effects of the system is.

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
        lv.append(LotkaVolterra2OND(param_set[0], param_set[1],
                                    param_set[2], param_set[3],
                                    param_set[5], param_set[6], 0))
        lv_trans.append(LotkaVolterra2OND(param_set[0], param_set[1],
                                          param_set[2], param_set[4],
                                          param_set[5], param_set[6], 0))

    times = []
    times_trans = []
    for i in range(len(lv)):
        times.append(np.linspace(0., max_time, num_times))
    for i in range(len(lv_trans)):
        times_trans.append(np.linspace(0., max_time, num_times))

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

    if range_cover:
        data, high, low = rangeCover(raw_data)
        return data, low, high

    else:
        return raw_data


def simDataAlt(params, max_time, num_times, overlay, stochastic_reps=None):
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
    if stochastic_reps is None:
        for param_set in params:
            lv.append(LotkaVolterraND(param_set[0], param_set[1], param_set[2], param_set[3], 0))
            lv_trans.append(LotkaVolterraND(param_set[0], param_set[1], 
                param_set[2], param_set[4], 0))
    else:
        for param_set in params:
            lv.append(LotkaVolterraSND(param_set[0], param_set[1], param_set[2],
                param_set[3], param_set[4], 0))
            lv_trans.append(LotkaVolterraSND(param_set[0], param_set[1], 
                param_set[2], param_set[3], param_set[5], 0))
 
    times = []
    times_trans = []
    for i in range(len(lv)):
        times.append(np.linspace(0., max_time, num_times))
    for i in range(len(lv_trans)):
        times_trans.append(np.linspace(0., max_time, num_times))

    if stochastic_reps is None:
        xs = []
        xs_trans = []
        out_of_range = True
        while out_of_range:
            for i, sys in enumerate(lv):
                temp = sys.check_xs(times[i])
            for i, sys in enumerate(lv_trans):
                temp_trans = ys.check_xs(times[i])
            if not (np.max(temp) < np.max(sys._k) * 2. and 
                    np.max(temp_trans) < np.max(sys._k) * 2.):
                # overrange
                times[i] = np.linspace(0., np.max(times[i]) / 2., num_times)
            elif not (np.max(temp) > np.max(sys._k / 4.) and
                    np.max(temp_trans) > np.max(sys._k / 4.)):
                # underrange
                times[i] = np.linspace(0., np.max(times[i]) * 1.3, 
                        num_times)
            elif not (np.all(np.isfinite(temp)) and 
                    np.all(np.isfinite(temp))):
                # probably overrange
                times[i] = np.linspace(0., np.max(times[i]) / 2., num_times)
            else:
                xs.append(temp)
                xs_trans.append(temp_trans)
                out_of_range = False
        
        raw_data = []
        for i in range(len(lv)):
            f = overlay(xs[i])
            f_trans = overlay(xs_trans[i])
            raw_data.append([f, f_trans])

    else:
        xs = []
        xs_trans = []
        for i, sys in enumerate(lv):
            temp = sys.check_xs(times[i])
            sys._x = sys._init_x
            for r in range(stochastic_reps):
                temp = np.vstack((temp,sys.check_xs(times[i])))
                sys._x = sys._init_x
            xs.append(temp)

        for i, sys in enumerate(lv_trans):
            temp = sys.check_xs(times[i])
            sys._x = sys._init_x
            for r in range(stochastic_reps):
                temp = np.vstack((temp,sys.check_xs(times[i])))
                sys._x = sys._init_x
            xs_trans.append(temp)

        raw_data = []
        for i in range(len(lv)):
            f = overlay(xs[i])
            f_trans = overlay(xs_trans[i])
            raw_data.append([f, f_trans])

    return raw_data



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
        delayed(randInitPopsSim)(params[0], params[1], params[2], iterations)
        for params in param_arr)
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
                the records for f and f' (f and f' have the same length), and 
                data[1] holds the records for g and g' (g and g' have the same
                length). E.g.
                array([[[ 0,  2,  8,  6,  4],
                    [99, 98, 97, 96, 95]],

                   [[ 1,  4,  2, 16, 32],
                    [89, 88, 87, 86, 85]]])
    """

    frames = []
    # highestLow will come to contain max([min(f), min(g)])
    highestLow = min(data[0][0])
    # lowestHigh will come to contain min([max(f), max(g)])
    lowestHigh = max(data[0][0])
    for tup in data:
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

    #    print(lowestHigh)
    #    print(highestLow)
    return output, lowestHigh, highestLow


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
        if type(tup[0]) == list and type(tup[1])==list:
            col1 = np.array(tup[0])[:, np.newaxis]
            col2 = np.array(tup[1])[:, np.newaxis]
            out.append(np.hstack((col1, col2)))
        elif type(tup[0]) == np.ndarray and type(tup[1]) == np.ndarray:
            col1 = tup[0]
            col2 = tup[1]
            out.append(np.hstack((col1, col2)))
        else:
            raise ValueError('Cannot convert tuples to blocks; wrong format.')

    return out


def resampleToUniform(data, low, high):
    """ Converts a list of 2-d numpy arrays to a list of 2-d arrays that have
    been resampled so that the values in the first column are uniformly
    distributed.
    """

    out = []
    for block in data:
        out.append(resample.uniform(block, bounds=[low, high]))

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
                                                                        y]).reshape(
            1, -1)))
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
    estimate joint probability densities, and outputs a list of trained sklearn KernelDensity
    objects.
    """
    densities = []
    for block in data:
        if block.shape[0] > block.shape[1]:
            block = block.T
        print('Block shape for kde = {}'.format(block.shape))
        if np.max(block.shape) < 10:
            pdf = lambda x, y: np.nan
            densities.append(pdf)
        else:
            try:
                kde = stats.gaussian_kde(block)
                pdf = lambda x, y, kde=kde: kde.evaluate(np.array([x,y]).reshape(2,1))
                densities.append(pdf)
            except:
#                print("in blocksToScipyDensities, reverting to fixed bw\n")
#                kde = stats.gaussian_kde(block.T,bw_method=0.1)
                pdf = lambda x, y: np.nan
                densities.append(pdf)
    return densities


def meanHellinger(func1, func2, x_range):
    def integrand(x):
        f1 = lambda y: func1(y, x)
        f2 = lambda y: func2(y, x)

        return HellingerDistance(f1, f2, x_range)


    out = quad(integrand, x_range[0], x_range[1], epsabs=1. * 10 ** (-6),
               limit=30)

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


def distanceH2D(densities, x_range=[-np.inf, np.inf],
                y_range=[-np.inf, np.inf]):
    """ Returns a distance matrx.
    """
    s = len(densities)
    dmat = np.zeros((s, s))

    for i in trange(s):
        for j in trange(i, s):
            dmat[i, j] = Hellinger2D(densities[i], densities[j], x_range[0],
                                     x_range[1], y_range[0], y_range[1])
            dmat[j, i] = dmat[i, j]

    return dmat


def energyDistanceMatrix(data):
    """ Returns a distance matrx.
    """
    s = len(data)
    dmat = np.zeros((s, s))

    for i in trange(s):
        for j in trange(i, s):
            dmat[i, j] = EnergyDistance(data[i], data[j])
            dmat[j, i] = dmat[i, j]

    return dmat


def AH_loop_function(i, j, tuples):
    data, high, low = rangeCover([tuples[i],tuples[j]])

    if low >= high:
        return [i, j, np.nan]

    blocks = tuplesToBlocks(data)
    
    for block in blocks:
        print(block.shape)

    x_min = []
    x_max = []
#    x_std = []
    y_min = []
    y_max = []
#    y_std = []
    for block in blocks:
        try:
            x_min.append(np.min(block[:,0]))
            x_max.append(np.max(block[:,0]))
#            x_std.append(np.std(block[:,0]))
            y_min.append(np.min(block[:,1]))
            y_max.append(np.max(block[:,1]))
#            y_std.append(np.std(block[:,1]))
        except:
            if block.shape[0] == 0:
                print("Block is empty.")
                return [i, j, np.nan]

#    x_std = np.max(x_std)
#    x_min = np.min(x_min) - x_std
#    x_max = np.max(x_max) + x_std
#    y_std = np.max(y_std)
#    y_min = np.min(y_min) - y_std
#    y_max = np.max(y_max) + y_std

    x_min = np.min(x_min)
    x_max = np.max(x_max)
    y_min = np.min(y_min)
    y_max = np.max(y_max)

    densities = blocksToScipyDensities(blocks)

    if i == j:
        assert densities[0](x_min, y_min) == densities[1](x_min, y_min)

    out = [i, j, Hellinger2D(densities[0], densities[1], x_min, x_max, y_min, y_max)]
    
    if i == j:
        print("self distance = {}".format(out))

    return out


def AveHellinger(tuples, free_cores=2):
    """ Given a list of tuples (f', f), returns a distance matrix.
    """
    s = len(tuples)
    dmat = np.zeros((s, s))

    cpus = max(cpu_count() - free_cores, 1)

#    for i in trange(s):
#        for j in trange(i + 1, s):
#            data, high, low = rangeCover([tuples[i],tuples[j]])
#
#            blocks = tuplesToBlocks(data)
#
#            rblocks = resampleToUniform(blocks, low, high)
#
#            densities = blocksToScipyDensities(rblocks)
#
#            dmat[i, j] = Hellinger2D(densities[0], densities[1], x_range[0],
#                                     x_range[1], y_range[0], y_range[1])
#            dmat[j, i] = dmat[i, j]

    out = Parallel(n_jobs=cpus,verbose=100)(delayed(AH_loop_function)(i,j,tuples) for i in range(s) for j in range(i, s))

#    for i in trange(s):
#        for j in trange(i, s):
#            dmat[i, j] = out[i * (s - i) + (j - i)]
#            dmat[j, i] = dmat[i, j]

    for cell in out:
        print(cell)
        dmat[cell[0], cell[1]] = dmat[cell[1], cell[0]] = cell[2]

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


def energy_loop_function(i, j, blocks):
    
    for block in blocks:
#        print(block.shape)

        if block.shape[0] == 0:
            print("Block is empty.")
            return [i, j, np.nan]

    out = [i, j, EnergyDistance(blocks[i], blocks[j])]
    
    if i == j:
        print("self distance = {}".format(out[2]))

    return out


def energyDistanceMatrixParallel(blocks, free_cores=2, verbose=0):
    """ Given a list of tuples (f', f), returns a distance matrix.
    """
    s = len(blocks)
    dmat = np.zeros((s, s))

    cpus = max(cpu_count() - free_cores, 1)

    out = Parallel(n_jobs=cpus,verbose=verbose)(delayed(energy_loop_function)(i,j,blocks) 
            for i in trange(s) for j in trange(i, s))

    for cell in out:
#        print cell
        dmat[cell[0], cell[1]] = dmat[cell[1], cell[0]] = cell[2]

    return dmat


def energyDistanceMatrix(data):
    """ Returns a distance matrx.
    """
    s = len(data)
    dmat = np.zeros((s, s))

    for i in trange(s):
        for j in trange(i, s):
            dmat[i, j] = EnergyDistance(data[i], data[j])
            dmat[j, i] = dmat[i, j]

    return dmat


