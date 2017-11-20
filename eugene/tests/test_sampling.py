# test_sampling.py

import eugene.src.auxiliary.sampling.resample as resample
import numpy as np
from scipy import stats


def test_uniform():
    data = np.random.normal(size=(1000,2))
    resampled_data = resample.uniform(data)
#    h = np.histogram(resampled_data[:,0])
    assert len(resampled_data) == 10 * len(data)

    # compute K-S test for data
    [D_data, p_val_data] = stats.kstest(data[:,0], 'norm')

    assert p_val_data > 0.05

    # compute K-S test to show resampled_data does _not_ fit a normal
    [D_resamp_norm, p_val_resamp_norm] = stats.kstest(resampled_data[:,0], 'norm')

    assert p_val_resamp_norm < 0.05

    # compute K-S test to show resampled_data is indistinguishable from uniform
    # distribution
    [D_resamp_uni, p_val_resamp_uni] = stats.kstest(resampled_data[:,0],
            'uniform')

    assert p_val_resamp_uni < 0.05

def test_gaussian():
    data = np.random.random_sample(size=(1000,2))
    resampled_data = resample.gaussian(data)
#    h = np.histogram(resampled_data[:,0])
    assert len(resampled_data) == 10 * len(data)

    # compute K-S test for data
    [D_data, p_val_data] = stats.kstest(data[:,0], 'uniform')

    assert p_val_data > 0.05

    # compute K-S test to show resampled_data does _not_ fit a uniform
    # distribution
    [D_resamp_norm, p_val_resamp_norm] = stats.kstest(resampled_data[:,0],
            'uniform')

    assert p_val_resamp_norm < 0.05

    # compute K-S test to show resampled_data is indistinguishable from normal
    # distribution
    [D_resamp_uni, p_val_resamp_uni] = stats.kstest(resampled_data[:,0], 
            'norm')

    assert p_val_resamp_uni < 0.05

