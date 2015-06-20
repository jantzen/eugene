# TestVABProcess.py

from VABClasses import *
from VABProcess import *
from random import *

def test_EmpiricalDeriv():
    data = []
    for i in range(5):
        data.append(i**2)
    assert EmpiricalDeriv(data) == 4

    data = []
    f = lambda x:x**2
    for i in range(-1,14):
        data.append(f(i*.25))
    assert EmpiricalDeriv(data,.25) == 3

    data = []
    f = lambda x:x**2
    for i in range(-1,14):
        data.append(f(i*.25)+gauss(0,.2))
    x=EmpiricalDeriv(data,.25)
    assert 2.5 < x < 3.5 


def test_FindParamVals():
    pass


def test_SymFunc():
    # set up a system, sensors, and actuators
    sys = VABSystemExpGrowth(1,2)
    tsensor = VABTimeSensor([])
    psensor = VABPopulationSensor([0,10**12])
    pact = VABPopulationActuator([0,10**12])

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
    out1 = SymFunc(interface, sigma_1, 1, 2, 0.5)
    out2 = SymFunc(interface, sigma_2, 1, 2, 0.5)

    print 'Out 1: {}'.format(out1) 
    print 'Out 2: {}'.format(out2)

    assert out1**2 < .0001 and out2**2 > .0001


def test_SymmetryGroup():
    pass


def test_randomOperation():
    #Create two functions
    function_1 = Function("c[0]*v[0]",1,1)
    function_2 = Function("c[0]+v[0]",1,1)
    
    #Do a random operation
    function_3 = randomOperation(function_1,function_2)
    
    #Ensure the output is what it should be
    assert function_3._function == "(c[0]*v[0])+(c[1]+v[0])" or function_3._function == "(c[0]*v[0])*(c[1]+v[0])" or function_3._function == "pow(c[0]*v[0],c[1]+v[0])"
    
    #Ensure that the inputs remain unchanged
    assert function_1._function == "c[0]*v[0]"
    assert function_2._function == "c[0]+v[0]"
    

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
    sys = VABSystemLogistic(100,8,1)
    tsensor = VABTimeSensor([])
    xsensor = VABLogisticSensor([0,700])
    xact = VABLogisticActuator([0,700])

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,xsensor)])
    actuators = dict([(2,xact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # build a seed generation
    #func = Function("c[0]",1,1)
    func = FunctionTree(Expression("c[0]*v[0]",1,1))
    current_generation = [func]

    # build a simple deck
    deck = [Expression("v[0]",0),Expression("c[0]",1),Expression("c[1]",1),Expression("c[2]",1),Expression("1",0)]

    # create range objects
    const_range = Range(0,100)

    # Start the Genetic Algorithm
    final_generation = GeneticAlgorithm(interface, current_generation, 1, 2, 10, 0.1, const_range, deck, 40, 4, 50, 0.2)

    print final_generation
    for function in final_generation:
        print "Function: {}; error: {}\n".format(function._expression.Evaluate(),function._error)
       

