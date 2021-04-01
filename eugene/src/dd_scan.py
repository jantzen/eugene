# file: dd_scan.py

from eugene.src.auxiliary.probability import EnergyDistance
from eugene.src.data_prep.clipping import *
from eugene.src.data_prep.fragment_timeseries import *
from eugene.src.data_prep.initial_conditions import *
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import numpy as np
from colorama import Fore, Style
import copy
import sys
import traceback
import pdb


class Scanner( object ):

    def __init__(self, timeseries, window_width=None, step_size=1, steps=10):

        self._data = timeseries

        if window_width is None:
            self._window_width = max(self._data.shape)/10
        else:
            self._window_width = window_width
        
        self._step_size = step_size
        self._steps = 10

    def start_scan(self, frags=100, reps=10, alpha=1., free_cores=1, verbose=25):
        pass


class RefScanner( Scanner ):

    def __init__(self, timeseries, reference=None,  window_width=None,
            step_size=1, steps=10):
        self._data = timeseries

        if reference is None:
            mm = self._data.shape[1] / 10
            self._ref = self._data[:,:mm]
        else:
            self._ref = reference

        if window_width is None:
            self._window_width = max(self._data.shape)/10
        else:
            self._window_width = window_width
        
        self._step_size = step_size
        self._steps = 10
        self._gpu = False

    def start_scan(self, frags=100, reps=10, alpha=1., free_cores=1, verbose=25):
        length = self._data.shape[1]
        w = self._window_width
        c = frags

        cpus = max(cpu_count() - free_cores, 1)

        tmp = Parallel(
                n_jobs=cpus,verbose=verbose,max_nbytes=1e6,temp_folder='/tmp'
                )(delayed(_scan_loop)(i,w,c,reps, alpha, self._data[:,i:(i+w)], 
                    self._ref, steps=self._steps) 
                    for i in range(0, length - 1 - w, self._step_size))
        self._scan = np.asarray(tmp)


class DiffScanner( Scanner ):

    def __init__(self, timeseries, window_width=None, lag=None, step_size=1,
            steps=10):
        self._data = timeseries

        if window_width is None:
            self._window_width = int(max(self._data.shape) / 10.)
        else:
            self._window_width = int(window_width)
        
        if lag is None:
            self._lag = int(self._window_width / 10.)
        else:
            self._lag = int(lag)
        
        self._step_size = step_size
        self._steps = int(steps)

    def start_scan(self, frags=100, reps=10, alpha=1., free_cores=1, verbose=25):
        length = self._data.shape[1]
        w = self._window_width
        c = frags
        g = self._lag

        cpus = max(cpu_count() - free_cores, 1)
        tmp = Parallel(
                n_jobs=cpus,verbose=verbose,max_nbytes=1e6,temp_folder='/tmp'
                )(delayed(_scan_loop)(i,w,c, reps, alpha, 
                    self._data[:,(i+w+g):(i+w+g+w)], self._data[:,i:(i+w)],
                    steps=self._steps) 
                    for i in range(0,length - 1 - (2 * w + g), self._step_size))
        self._scan = np.asarray(tmp)


def _distance(untrans, trans, steps):
    untrans1 = copy.deepcopy(untrans[0])
    untrans2 = copy.deepcopy(untrans[1])
    trans1 = copy.deepcopy(trans[0])
    trans2 = copy.deepcopy(trans[1])

    # clip the data
    tmp_untrans1, tmp_untrans2 = clip_segments(untrans1, untrans2,
            steps)
    tmp_trans1, tmp_trans2 = clip_to_match(tmp_untrans1, tmp_untrans2, trans1,
            trans2)
    data1, data2 = zip_curves(tmp_untrans1, tmp_untrans2, tmp_trans1,
            tmp_trans2)
    dist = EnergyDistance(data1.T, data2.T)

    return dist


def _scan_loop(i, w, c, reps, alpha, data, baseline, steps):
    segment = data

    reps = int(reps)

    # split into trans and untrans components
    split_data = split_timeseries([baseline, segment], 2*c)

    try:
        untrans, trans = choose_untrans_trans(split_data, reps,
                alpha=alpha)

    except Exception as inst:
        print("Error choosing trans, untrans: ") 
        print(inst)
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        d = np.nan

    try:
        d = _distance(untrans, trans, steps)

    except Exception as inst:
        print("Error computing distance: ")
        print(inst)
        d = np.nan

    print(Fore.RED + 'i = {}, d = {}.'.format(i, d))
    print(Style.RESET_ALL)

    return [i, d]

