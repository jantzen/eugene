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


def test_surface_fit():
    xdata = [np.linspace(0,10,100)]
    ydata = 10. - 2. * xdata[0] + 0.5 * xdata[0]**2

    params, cov = eu.compare.surface_fit(xdata, ydata, 2)

    assert (params -  np.array([10., -2., 0.5]) < 10**(-5)).all()

    xdata = [np.random.rand(100,) * 10., np.random.rand(100,) * 10.]
    ydata = (212. + 1.5 * xdata[1] + pow(xdata[1], 2) + 3. * xdata[0] - 5. *
        xdata[0] * xdata[1] - 2.2 * pow(xdata[0],2)) 

    params, cov = eu.compare.surface_fit(xdata, ydata, 2)


    assert (params - np.array([212., 1.5, 1., 3., -5., -2.2]) < 10**(-5)).all()


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

def test_BuildSymModel():
    # build sensors and actuators
    noise_stdev = 10**(-4)
    resolution = [300, 1]

    tsensor = eu.sensors.VABTimeSensor([])
    xsensor0 = eu.sensors.CCVoltageSensor([-10**23,10.**23], 0, noise_stdev, False)
    xsensor1 = eu.sensors.CCVoltageSensor([-10**23,10.**23], 1, noise_stdev, False)
    xsensor2 = eu.sensors.CCVoltageSensor([-10**23,10.**23], 2, noise_stdev, False)
    xact0 = eu.actuators.CCVoltageActuator([0.,10.**23], 0)
    xact1 = eu.actuators.CCVoltageActuator([0.,10.**23], 0)
    xact2 = eu.actuators.CCVoltageActuator([0.,10.**23], 0)
    tact = eu.actuators.VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(3, tsensor), (0, xsensor0), (1, xsensor1), (2, xsensor2)])
    actuators = dict([(3,tact),(0,xact0), (1,xact1), (2,xact2)])

    # build system 
    sys = ChaoticCircuit(5)

    # build corresponding interfaces
    interface = eu.interface.VABSystemInterface(sensors, actuators, sys)

    # build ROI
    ROI = dict([(3,[0., 10.]), (0, np.array([0.,-9.22617])), (1,
        np.array([0.,3.86821])), (2, np.array([0.,2.60234]))])
            
    # collect data
    data = eu.interface.TimeSampleData(3, [0,1,2], interface,
            ROI, resolution, target_value_points=True)

    model = eu.compare.BuildSymModel(data, 3, [0,1,2],0,0.1)

    return data, model
