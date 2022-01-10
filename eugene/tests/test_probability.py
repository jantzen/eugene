# test_probability.py
from eugene.src.auxiliary.probability import *
import numpy as np


def test_EnergyDistance():
    # 1-D
    x1 = np.random.normal(size=(1000, 1))
    x2 = np.random.normal(size=(1000, 1))
    y1 = np.random.normal(loc=2.0, size=(1000,1))
    y2 = np.random.normal(loc=2.0, size=(1000,1))

    x1x2 = EnergyDistance(x1, x2)
    x1y1 = EnergyDistance(x1, y1)
    y1y2 = EnergyDistance(y1, y2)
    x2y2 = EnergyDistance(x2, y2)

    assert x1x2 < x1y1 and x1x2 < x2y2
    assert y1y2 < x1y1 and y1y2 < x2y2

    # 5-D
    x1 = np.random.normal(size=(1000, 5))
    x2 = np.random.normal(size=(1000, 5))
    y1 = np.random.normal(loc=2.0, size=(1000,5))
    y2 = np.random.normal(loc=2.0, size=(1000,5))

    x1x2 = EnergyDistance(x1, x2)
    x1y1 = EnergyDistance(x1, y1)
    y1y2 = EnergyDistance(y1, y2)
    x2y2 = EnergyDistance(x2, y2)

    assert x1x2 < x1y1 and x1x2 < x2y2
    assert y1y2 < x1y1 and y1y2 < x2y2

    # 1-D gpu
    x1 = np.random.normal(size=(1000, 1))
    x2 = np.random.normal(size=(1000, 1))
    y1 = np.random.normal(loc=2.0, size=(1000,1))
    y2 = np.random.normal(loc=2.0, size=(1000,1))

    x1x2 = EnergyDistance(x1, x2, gpu=True)
    x1y1 = EnergyDistance(x1, y1, gpu=True)
    y1y2 = EnergyDistance(y1, y2, gpu=True)
    x2y2 = EnergyDistance(x2, y2, gpu=True)

    assert x1x2 < x1y1 and x1x2 < x2y2
    assert y1y2 < x1y1 and y1y2 < x2y2


    # 5-D gpu
    x1 = np.random.normal(size=(1000, 5))
    x2 = np.random.normal(size=(1000, 5))
    y1 = np.random.normal(loc=2.0, size=(1000,5))
    y2 = np.random.normal(loc=2.0, size=(1000,5))

    x1x2 = EnergyDistance(x1, x2, gpu=True)
    x1y1 = EnergyDistance(x1, y1, gpu=True)
    y1y2 = EnergyDistance(y1, y2, gpu=True)
    x2y2 = EnergyDistance(x2, y2, gpu=True)

    assert x1x2 < x1y1 and x1x2 < x2y2
    assert y1y2 < x1y1 and y1y2 < x2y2


def test_ksample():
    x1 = np.random.normal(size=(1000, 5))
    y1 = np.random.normal(loc=2.0, size=(1000,5))
    y2 = np.random.normal(loc=2.0, size=(1000,5))
    x2 = np.random.normal(size=(1000, 5))

    x1x2 = EnergyDistance(x1, x2, gpu=True)
    x1y1 = EnergyDistance(x1, y1, gpu=True)
    x1y2 = EnergyDistance(x1, y2, gpu=True)
    x2y1 = EnergyDistance(x2, y1, gpu=True)
    x2y2 = EnergyDistance(x2, y2, gpu=True)
    y1y2 = EnergyDistance(y1, y2, gpu=True)
    assert np.abs(kSample(x1, x2) - x1x2) <= 10.**(-12)
    assert np.abs(kSample(x1, x2, y1, y2) - (x1x2 + x1y1 + x1y2 + x2y1 
            + x2y2 + y1y2)) <= 10.**(-12)


def test_nd_gaussian_pdf():
    points = np.arange(4, dtype=float).reshape(1, -1)
    densities = nd_gaussian_pdf(0., 1., points)
    assert np.abs(densities[0,0] - 0.398942) < 10.**-5
    assert np.abs(densities[0,1] - 0.241971) < 10.**-5
    assert np.abs(densities[0,2] - 0.05399) < 10.**-5
    assert np.abs(densities[0,3] - 0.00443185) < 10.**-5


def test_significant():
    with warnings.catch_warnings(record=True) as w:
        # test with 3-D data
        mean1 = np.ones(3)
        mean2 = 2. * mean1 
    
        cov1 = np.array([[1., 0.2, 0.3],[0.2, 1., 0.8],[0.3, 0.8, 1.]])
        cov2 = np.array([[2., 0.4, 0.1],[0.4, 2., 0.2],[0.1, 0.2, 2.]])
    
        sample1 = np.random.multivariate_normal(mean1, cov1, size = 1000)
        sample2 = np.random.multivariate_normal(mean1, cov1, size = 1000)
        sample3 = np.random.multivariate_normal(mean2, cov1, size = 1000)
        sample4 = np.random.multivariate_normal(mean1, cov2, size = 1000)
    
        assert not significant(sample1, sample2, 
    	    EnergyDistance(sample1, sample2))
        assert significant(sample1, sample3, 
    	    EnergyDistance(sample1, sample3))
        assert significant(sample1, sample4, 
    	    EnergyDistance(sample1, sample4))
    
        # test with 1-D data
        mean1 = 1.
        mean2 = 2.
    
        sigma1 = 0.5
        sigma2 = 0.25
    
        sample1 = np.random.normal(mean1, sigma1, size = 1000).reshape(-1,1)
        sample2 = np.random.normal(mean1, sigma1, size = 1000).reshape(-1,1)
        sample3 = np.random.normal(mean2, sigma1, size = 1000).reshape(-1,1)
        sample4 = np.random.normal(mean1, sigma2, size = 1000).reshape(-1,1)
    
        assert not significant(sample1, sample2, 
    	    EnergyDistance(sample1, sample2))
        assert significant(sample1, sample3, 
    	    EnergyDistance(sample1, sample3))
        assert significant(sample1, sample4, 
    	    EnergyDistance(sample1, sample4))
    
        sample1 = np.random.rayleigh(1., size = 1000).reshape(-1,1)
        sample2 = np.random.rayleigh(1., size = 1000).reshape(-1,1)
        sample3 = np.random.rayleigh(2., size = 1000).reshape(-1,1)
        sample4 = np.random.rayleigh(1., size = 10000).reshape(-1,1)
        sample5 = np.random.rayleigh(1.3, size = 10000).reshape(-1,1)
    
        assert not significant(sample1, sample2, 
    	    EnergyDistance(sample1, sample2))
        assert significant(sample1, sample3, 
    	    EnergyDistance(sample1, sample3))
        assert significant(sample4, sample5, 
    	    EnergyDistance(sample4, sample5), n=1000)
    
        sample1 = np.random.triangular(0., 1., 2., size = 1000).reshape(-1,1)
        sample2 = np.random.triangular(0., 1., 2., size = 1000).reshape(-1,1)
        sample3 = np.random.triangular(1., 2., 3., size = 1000).reshape(-1,1)
        sample4 = np.random.triangular(-1., 1., 3., size = 1000).reshape(-1,1)
    
        assert not significant(sample1, sample2, 
    	    EnergyDistance(sample1, sample2))
        assert significant(sample1, sample3, 
    	    EnergyDistance(sample1, sample3))
        assert significant(sample1, sample4, 
    	    EnergyDistance(sample1, sample4))
    
        sample1 = np.random.normal(mean1, sigma1, size = 10000).reshape(-1,1)
        sample2 = np.random.rayleigh(1., size = 10000).reshape(-1,1)
        sample3 = np.random.triangular(0., 1., 2., size = 10000).reshape(-1,1)
    
        assert significant(sample1, sample2, 
    	    EnergyDistance(sample1, sample2), n=1000)
        assert significant(sample1, sample3, 
    	    EnergyDistance(sample1, sample3), n=1000)
        assert significant(sample3, sample2, 
    	    EnergyDistance(sample3, sample2), n=1000)

    for warn in w:
        print(warn.message)
