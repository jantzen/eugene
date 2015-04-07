# TestVABProcess.py

from VABClasses import *
from VABProcess import *


def test_SymFunc():
    # set up a system, sensors, and actuators
    sys = VABSystemExpGrowth(1,0.2)
    tsensor = VABTimeSensor()
    psensor = VABPopulationSensor()
    pact = VABPopulationActuator()

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,psensor)])
    actuators = dict([(2,pact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # define the transformation function of interest
    def sigma(x):
        return 2*x

    # call SymFunc to test whether sigma is a symmetry
    out = SymFunc(interface, sigma, 1, 2, 0.5, 1, 1)

    assert out

def test_SymFunc_negative():
    # set up a system, sensors, and actuators
    sys = VABSystemExpGrowth(1,0.2)
    tsensor = VABTimeSensor()
    psensor = VABPopulationSensor()
    pact = VABPopulationActuator()

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,psensor)])
    actuators = dict([(2,pact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # define the transformation function of interest
    def sigma(x):
        return x+20

    # call SymFunc to test whether sigma is a symmetry
    out = SymFunc(interface, sigma, 1, 2, 0.5, 1, 1)

    assert not out


def test_GeneticAlgorithm():
    pass
