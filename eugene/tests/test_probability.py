# test_probability.py
from eugene.src.auxiliary.probability import *
from eugene.src.energy_test_stat import *
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
    
    # Significance
    assert significant(x1, y1, D=x1y1, n=50)
    assert not significant(x1, x2, D=x1x2, n=50)


def test_nd_gaussian_pdf():
    points = np.arange(4, dtype=float).reshape(1, -1)
    densities = nd_gaussian_pdf(0., 1., points)
    assert np.abs(densities[0,0] - 0.398942) < 10.**-5
    assert np.abs(densities[0,1] - 0.241971) < 10.**-5
    assert np.abs(densities[0,2] - 0.05399) < 10.**-5
    assert np.abs(densities[0,3] - 0.00443185) < 10.**-5


def testSigma2():
    X=[1,1,1,1]
    Y=[1,1,1,5]
    expect=(5-1)*len(Y)
    assert sigma2(X,Y)==expect


def testTwoSampleToy():
    X=[1,1,1,1]
    Y=[1,1,1,5]
    assert twoSample(X,Y) == kSample(X,Y)


def testTwoSampleProper():
    # Expected value of energy distance between norm(0,1) and norm(2,1)
    norm0norm2_distances = []
    for i in range(1000):
        A = np.random.normal(size=200)
        B = np.random.normal(size=200)
        norm0norm0 = twoSample(A,B)

        C = np.random.normal(loc=2, size=200)
        D = np.random.normal(loc=2, size=200)
        norm2norm2 = twoSample(C,D)
        
        norm0norm2 = twoSample(A,C)
        
        norm0norm2_distances.append(norm0norm2)
        
        # It would be very weird if this were not so
        assert norm0norm2 > norm0norm0

    # E(X_i - Y_m) = 1; E(X_i - X_j) = 0 = E(Y_l - Y_m)
    # average energy distance should be very near 200?
    avgen = (sum(norm0norm2_distances)/len(norm0norm2_distances))
    assert avgen == 200 # this will not pass but we'll see how close

