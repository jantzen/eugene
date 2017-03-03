import eugene as eu
import numpy as np
import pdb
from eugene.src.virtual_sys.chaotic_circuits import *

def test_exponents():
    assert eu.compare.exponents(1, 2) == [[0],[1],[2]]
    assert eu.compare.exponents(2, 2) == [[0,0],[0,1],[0,2],[1,0],[1,1],[2,0]]


def test_npoly_val():
    exponents1 = eu.compare.exponents(1,2)
    exponents2 = eu.compare.exponents(2,2)
    params1 = [2, 2, 2]
    params2 = [1, 1, 1, 1, 1, 1]
    assert eu.compare.npoly_val(params1, exponents1, [2]) == 2*2**0 + 2*2**1 + 2*2**2
    assert eu.compare.npoly_val(params2, exponents2, [1, 2]) == (1 + 1*1**0*2**1 +
            1*1**0*2**2 + 1*1**1*2**0 + 1*1**1*2**1 + 1*1**2*2**0)


def test_residuals():
    exponents1 = eu.compare.exponents(1,2)
    exponents2 = eu.compare.exponents(2,2)
    params1 = [2, 2, 2]
    params2 = [1, 1, 1, 1, 1, 1]

    xdata1 = [np.array([1, 2, 3])]
    xdata2 = [np.array([1, 2, 3]), np.array([1, 2, 3])]

    # also create a 2D array to test input modes
    xdata3 = np.hstack((np.array([1,2,3]).reshape(3,1),
        np.array([1,2,3]).reshape(3,1)))

    ydata1a = []
    ydata1b = []
    for i, x in enumerate(xdata1[0]):
        y = 0
        for j, p in enumerate(params1):
            y = y + p * pow(x, exponents1[j][0])
        ydata1a.append(y)
        ydata1b.append(y - 1)
    ydata1a = np.array(ydata1a)
    ydata1b = np.array(ydata1b)
    
    assert eu.compare.residuals(params1, exponents1, xdata1, ydata1a) == [0, 0,
            0]
    assert eu.compare.residuals(params1, exponents1, xdata1, ydata1b) == [-1, -1,
            -1]

    ydata2 = []
    for i in range(len(xdata2[0])):
            y = 0
            for j, p in enumerate(params2):
                term = p
                for var in range(len(xdata2)):
                    term = term * pow(xdata2[var][i], exponents2[j][var])
                y += term
            ydata2.append(y)
    ydata2 = np.array(ydata2)

    assert eu.compare.residuals(params2, exponents2, xdata2, ydata2) == [0, 0,
        0]

    assert eu.compare.residuals(params2, exponents2, xdata3, ydata2) == [0, 0,
            0]


def test_surface_fit():
    xdata = [np.linspace(0,10,50)]
    ydata = 37. - 1.37 * xdata[0]

    params = eu.compare.surface_fit(xdata, ydata, 1)

    assert (params - np.array([37., -1.37]) < 10**(-5)).all()

    xdata = [np.linspace(0,10,100)]
    ydata = 10. - 2. * xdata[0] + 0.5 * xdata[0]**2

    params = eu.compare.surface_fit(xdata, ydata, 2)

    assert (params -  np.array([10., -2., 0.5]) < 10**(-5)).all()

    xdata = [np.random.rand(100,) * 10., np.random.rand(100,) * 10.]
    ydata = (212. + 1.5 * xdata[1] + pow(xdata[1], 2) + 3. * xdata[0] - 5. *
        xdata[0] * xdata[1] - 2.2 * pow(xdata[0],2)) 

    params = eu.compare.surface_fit(xdata, ydata, 2)


    assert (params - np.array([212., 1.5, 1., 3., -5., -2.2]) < 10**(-5)).all()



def test_surface_fit_sparse():
    xdata = [np.linspace(0,10,20)]
    ydata = 10. - 2. * xdata[0] + 0.5 * xdata[0]**2

    params = eu.compare.surface_fit(xdata, ydata, 2)

    assert (abs(params -  np.array([10., -2., 0.5])) < 10**(-5)).all()

    xdata = [np.random.rand(100,) * 10., np.random.rand(100,) * 10.]
    ydata = (212. + 1.5 * xdata[1] + pow(xdata[1], 2) + 3. * xdata[0] - 5. *
        xdata[0] * xdata[1] - 2.2 * pow(xdata[0],2)) 

    params = eu.compare.surface_fit(xdata, ydata, 2)


    assert (abs(params - np.array([212., 1.5, 1., 3., -5., -2.2]) < 10**(-5))).all()



def test_FitPolyCV():
    xdata = np.linspace(0,10,100).reshape(100,1)
    ydata = 10. - 2. * xdata + 0.5 * xdata**2
    data = np.hstack((xdata, ydata))
    params = eu.compare.FitPolyCV(data)[0]
    assert (abs(params - np.array([10., -2., 0.5])) < 10**(-5)).all()

    x0 = np.random.rand(100,1) * 10.
    x1 = np.random.rand(100,1) * 10.
    y = (1. + 13. * x1 - 17. * pow(x1, 2) + 0.1 * x0 - 100 * x0 * x1 + 5. * pow(x0,
        2))
    data = np.hstack((x0, x1, y))
    params = eu.compare.FitPolyCV(data)[0]
    assert (abs(params - np.array([1., 13., -17., 0.1, -100., 5.])) < 10**(-5)).all()
