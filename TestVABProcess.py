# TestVABProcess.py

from VABClasses import *
from VABProcess import *


def test_SymFunc():
    # set up a system, sensors, and actuators
    sys = VABSystemExpGrowth(1,2)
    tsensor = VABTimeSensor()
    psensor = VABPopulationSensor()
    pact = VABPopulationActuator()

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,psensor)])
    actuators = dict([(2,pact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # define the transformation function of interest
    sigma_1 = Function("c[0]*v[0]",1,1)
    sigma_1.SetConstants([2])
 
    sigma_2 = Function("c[0]+v[0]",1,1)
    sigma_2.SetConstants([2])


    # call SymFunc to test whether sigma is a symmetry
    out1 = SymFunc(interface, sigma_1, 1, 2, 0, 0.5)
    out2 = SymFunc(interface, sigma_2, 1, 2, 0, 0.5)

    print 'Out 1: {}'.format(out1) 
    print 'Out 2: {}'.format(out2)

    assert out1 < .01 and out2 > .01


def test_randomOperation():
    pass


def test_GeneticAlgorithm():
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

    # build a seed generation
    func = Function("c[0]",1,1)
    current_generation = [func]

    # build a simple deck
    deck = [Function("v[1]",0,1),Function("c[0]",1,0)]

    # create range objects
    const_range = Range(-100,100)

    # Start the Genetic Algorithm
    GeneticAlgorithm(interface, current_generation, 0, 1, 10, 0.1, [const_range], deck, 2, 2, 10, 10)

    print current_generation
