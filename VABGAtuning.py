# VABGAtuning.py
""" 
    Module for testing the various genetic algorithm implementations in 
    TestVABProcess.py. These methods can be called with a variety of
    arguments allow for command-line interactive experimentation.
"""
from VABClasses import *
from VABProcess import *
from random import *
import copy


def test_GeneticAlgorithm():
    # set up a system, sensors, and actuators
    sys = VABSystemExpGrowth(1,0.002)
    tsensor = VABTimeSensor([])
    psensor = VABPopulationSensor([0,700])
    pact = VABPopulationActuator([0,700])

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,psensor)])
    actuators = dict([(2,pact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # build a seed generation
    #func = Function("c[0]",1,1)
    expr = Expression("",0,0)
    expr.SetLeft(Expression("c[0]",0,1))
    expr.SetTerminal("*")
    expr.SetRight(Expression("v[0]",1,0))
    func = FunctionTree(expr)
    current_generation = [func]

    # build a simple deck
    deck = [Expression("v[0]",1,0),Expression("c[0]",0,1),Expression("c[1]",0,1)]

    # create range objects
    const_range = Range(0,1)

    # Start the Genetic Algorithm
    final_generation = GeneticAlgorithm(interface, current_generation, 1, 2, 10, 0.1, const_range, deck, 10, 4, 10, 0.3)

    # print final_generation
    print "\n\nFINAL GENERATION:  \n"
    for function in final_generation:
        print "Function: {}; error: {}\n".format(function._expression.Evaluate(),function._error)
       


def test_GeneticAlgorithmAndLogistic():
    # set up a system, sensors, and actuators
    sys = VABSystemLogisticVirtual(100,8,1)
    tsensor = VABTimeSensor([0,10**12])
    xsensor = VABLogisticSensorVirtual([0,700])
    xact = VABLogisticActuator_X([0,700])
    tact = VABLogisticActuator_T([0,10**12])

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,xsensor)])
    actuators = dict([(1,tact),(2,xact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # build a seed generation
    #func = Function("c[0]",1,1)
#    func = FunctionTree(Expression("c[0]*v[0]",1,1))
    expr = Expression("",0,0)
    expr.SetLeft(Expression("c[0]",0,1))
    expr.SetTerminal("*")
    expr.SetRight(Expression("v[0]",1,0))
    func = FunctionTree(expr)
    current_generation = [func]

    # build a simple deck
    deck = [Expression("v[0]",0),Expression("c[0]",1),Expression("c[1]",1),Expression("c[2]",1),Expression("1",0)]

    # create range objects
    const_range = Range(0,100)

    # Start the Genetic Algorithm
    final_generation = GeneticAlgorithm(interface, current_generation, 1, 2, 10, 0.1, const_range, deck, 50, 10, 50, 0.2)

    print final_generation
    for function in final_generation:
        print "Function: {}; error: {}\n".format(function._expression.Evaluate(),function._error)



def test_GeneticAlgorithmAndLogisticVirtual(cores=1, inductive_threshold=10,
        time_interval=0.2, generation_limit=100, num_mutes=10,
        generation_size=1000, percent_guaranteed=0.1,
        ROI=dict([(1,[0.5,1.5]),(2,[1,100])])):

    # set up a system, sensors, and actuators
    sys = VABSystemLogisticVirtual(100,8,1)

    xsensor = VABLogisticSensorVirtual([1,700])
    tsensor = VABTimeSensorVirtual([0,10**12])
    xact = VABLogisticActuator_X([1,1000])
    tact = VABLogisticActuator_T([0,10**12])

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,xsensor)])
    actuators = dict([(1,tact),(2,xact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)
    interfaces = []
    for i in range(cores):
        interfaces.append(copy.deepcopy(interface))

    # build a seed generation
    expr1 = Expression("")
    expr1.SetLeft(Expression("c[0]",set([]),set([0])))
    expr1.SetTerminal("*")
    expr1.SetRight(Expression("v[0]",set([0]),set([])))
    func1 = FunctionTree(expr1)
    expr2 = Expression("")
    expr2.SetLeft(Expression("c[0]",set([]),set([0])))
    expr2.SetTerminal("+")
    expr2.SetRight(Expression("v[0]",set([0]),set([])))
    func2 = FunctionTree(expr2)
 
    seed_generation = [func1,func2]

    # build a simple deck
    deck = [Expression("v[0]",set([0]),set([])),Expression("c[0]",set([]),set([0])),Expression("1"),Expression("5"),Expression("10"),Expression("50"),Expression("100"),Expression("500"),Expression("1000")]

    # create range objects
#    const_range = Range(0,100)

    # Start the Genetic Algorithm
    final_generation = GeneticAlgorithm(cores, interfaces, seed_generation, 1,
            2, inductive_threshold, time_interval, ROI, deck, generation_limit, num_mutes, generation_size, percent_guaranteed)

    print final_generation
    for function in final_generation:
        print "Function: {}; error: {}\n".format(function._expression.Evaluate(),function._error)
 
