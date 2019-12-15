# fragment_timeseries.py

import warnings
import numpy as np

""" Methods for dividing a timeseries into multiple sub-series for anaylsis by
dynamical distance.
"""

def split_timeseries(data, num_frags):
    """ data: a list of length n of (vars x sample)  numpy arrays, presumed 
    to be timeseries describing n different systems or treatments.

    returns: an n-length list of num_frags length lists of numpy arrays
    """
    ll = data[0].shape[1]
    # data verification
    for sys in data:
        # verify format
        if sys.shape[0] > sys.shape[1]:
            errmsg = 'Some timeseries data appears to be transposed.'
            warnings.warn(errmsg)
        # check whether all timeseries are the same length
        if not sys.shape[1] == ll:
            errmsg = """Some timeseries are of different lengths.
            This will result in framgents of different lengths after splitting
            each timeseries."""
            warnings.warn(errmsg)

    # split timeseries
    fragmented_data = []
    for sys in data:
        tmp = np.array_split(sys, num_frags, axis=1)

        # ensure all curves are of the same length
        if (not tmp[0].shape == tmp[-1].shape):
            # find the minimum length
            m = tmp[-1].shape[1]
            print("Trimming all sub-series to length {}.".format(m))
            data_split = []
            for curve in tmp:
                data_split.append(curve[:,:m])
        else:
            data_split = tmp

        fragmented_data.append(data_split)

    return fragmented_data