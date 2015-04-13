# TestVABClasses.py

from VABClasses import *

def test_sensors():
    sys = VABSystemExpGrowth(1, 0.2)
    tsensor = VABTimeSensor([])
    t = tsensor.read(sys)
    ttest = t == sys._time
    psensor = VABPopulationSensor([0,10**12])
    p = psensor.read(sys)
    ptest = p == sys._population

    assert ttest and ptest

    sys = VABSystemExpGrowth(1, 20)
    tsensor = VABTimeSensor([])
    psensor = VABPopulationSensor([0,10**12])
    p = psensor.read(sys)
    ptest = p == 'OutofRange'

 
def test_actuators():
    sys = VABSystemExpGrowth(1, 0.2)
    pact = VABPopulationActuator([])
    pact.set(sys,25)
    
    assert 25 == sys._population
  
   
def test_growth_model():   
    sys1 = VABSystemExpGrowth(1, 10)
    sys2 = VABSystemExpGrowth(5, 10)

    psensor = VABPopulationSensor([0,10**12])

    time.sleep(0.02)
 
    p1 = psensor.read(sys1)
    p2 = psensor.read(sys2)
    
    print(5*p1)
    print(p2)

    assert (5*p1 - .1) < p2 and p2 < (5*p1 + .1)


def test_VABSigmoidSystem():
    pass


def test_Function_init():
    """Ensure that internal constants are set correctly
    given a passed function string
    """
    func1 = Function("c[0]*v[0]+c[1]", 2, 1)
    
    assert func1._const_count == 2


def test_Function_SetConstants():
    func1 = Function("c[0]*v[0]+c[1]", 2, 1)
    func1.SetConstants([37,59])
    flag=0

    try:
        func1.SetConstants([1,2,3])
    except:
        flag=1

    assert func1._constants == [37,59] and flag == 1
    

def test_Function_EvaluateAt():
    func = Function("c[0]*v[0]+c[1]",2,1)
    func.SetConstants([2,1])

    assert func.EvaluateAt([1]) == 3


def test_IncrementIndices():
    """The purpose of the function being tested is to change the constant indices
    in one function so that they do not comflict with those
    in another with which it is being combined.
    """
    func = Function("c[0]+c[1]+c[2]*v[0]",3,1)
    func.IncrementIndices(3)

    assert func._const_count == 3 
    assert func._function == "c[3]+c[4]+c[5]*v[0]"

    func.IncrementIndices(10)
    
    assert func._var_count == 1
    assert func._function == "c[13]+c[14]+c[15]*v[0]"


def test_Function_MultiplyAddPower():

    func1 = Function("v[0]",0,1)
    func2 = Function("c[0]*v[0]",1,1)
    func3 = Function("c[0]*v[0]+c[1]",2,1)
    
    func1.Multiply(func2)
    func2.Add(func3)
    func3.Power(func3)

    assert func1._function == "(v[0])*(c[0]*v[0])"
    assert func2._function == "(c[0]*v[0])+(c[1]*v[0]+c[2])"
    assert func3._function == "pow(c[0]*v[0]+c[1],c[2]*v[0]+c[3])"


def test_VABSystemInterface():
    # set up a system, sensors, and actuators
    sys = VABSystemExpGrowth(1,2)
    tsensor = VABTimeSensor([])
    psensor = VABPopulationSensor([0,10**12])
    pact = VABPopulationActuator([0,10**4])

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,psensor)])
    actuators = dict([(2,pact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # make sure one can pull the range of a sensor
    assert interface.get_sensor_range(2) == [0,10**12]
