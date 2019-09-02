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


def test_nd_gaussian_pdf():
    points = np.arange(4, dtype=float).reshape(1, -1)
    densities = nd_gaussian_pdf(0., 1., points)
    assert np.abs(densities[0,0] - 0.398942) < 10.**-5
    assert np.abs(densities[0,1] - 0.241971) < 10.**-5
    assert np.abs(densities[0,2] - 0.05399) < 10.**-5
    assert np.abs(densities[0,3] - 0.00443185) < 10.**-5

