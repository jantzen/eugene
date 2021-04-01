# test_fragment_time_series.py

import warnings
from eugene.src.data_prep.fragment_timeseries import *

def test_split_timeseries():
    # cut, evenly divisible
    timeseries = [np.ones((3,5000)), np.ones((3,5000))]
    timeseries_split = split_timeseries(timeseries, 500)
    
    assert len(timeseries_split) == 2
    assert len(timeseries_split[0]) == 500
    assert len(timeseries_split[1]) == 500
    assert timeseries_split[0][0].shape[1] == 10
    assert timeseries_split[0][0].shape == timeseries_split[0][1].shape

    # cut, not evenly divisible
    timeseries = [np.ones((3,5000)), np.ones((3,5000))]
    timeseries_split = split_timeseries(timeseries, 333)

    assert len(timeseries_split) == 2
    assert len(timeseries_split[0]) == 333
    assert timeseries_split[0][0].shape[1] == 15
    assert timeseries_split[0][0].shape == timeseries_split[0][1].shape

    # verify warnings trigger as expected
    with warnings.catch_warnings(record=True) as w:
        timeseries = [np.ones((5000,3))]
        timeseries_split = split_timeseries(timeseries, 500)

        timeseries = [np.ones((3,5000)), np.ones((3,3000))]
        timeseries_split = split_timeseries(timeseries, 500)

    assert len(w) == 2

def test_fixed_length_frags():

    # cut, evenly divisible
    timeseries = [np.ones((3,5000)), np.ones((3,5000))]
    timeseries_split = fixed_length_frags(timeseries, 500)
    
    assert len(timeseries_split) == 2
    assert len(timeseries_split[0]) == 10
    assert len(timeseries_split[1]) == 10
    assert timeseries_split[0][0].shape[1] == 500
    assert timeseries_split[0][0].shape == timeseries_split[0][9].shape

    # cut, not evenly divisible
    timeseries = [np.ones((3,5000)), np.ones((3,5000))]
    timeseries_split = fixed_length_frags(timeseries, 333)

    assert len(timeseries_split) == 2
    assert len(timeseries_split[0]) == 16
    assert timeseries_split[0][0].shape[1] == 333
    assert timeseries_split[0][0].shape == timeseries_split[0][14].shape
    assert timeseries_split[0][-1].shape[1] == 5

    # verify warnings trigger as expected
    with warnings.catch_warnings(record=True) as w:
        timeseries = [np.ones((5000,3))]
        timeseries_split = fixed_length_frags(timeseries, 500)

    assert len(w) == 1

