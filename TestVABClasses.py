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
    
    # print(5*p1)
    # print(p2)

    assert (5*p1 - .1) < p2 and p2 < (5*p1 + .1)


def test_Virtual_interface():
#    pdb.set_trace()
    sys = VABSystemLogisticVirtual(100,8,1)

    xsensor = VABLogisticSensorVirtual([0,10**12])
    tsensor = VABTimeSensorVirtual([0,10**12])
    xact = VABLogisticActuator_X([1,1000])
    tact = VABLogisticActuator_T([0,10**12])

    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,xsensor)])
    actuators = dict([(1,tact),(2,xact)])

    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)

    # set the time
    interface.set_actuator(1,0.2)

    # read the time and population
    time = interface.read_sensor(1)
    pop = interface.read_sensor(2)

    assert time == 0.2 and pop == 100*1/((100-1)*math.exp(-8*0.2)+1)


def test_VABSystemLogistic():
    K = 100
    c = 1
    r = 2
    x0 = 1
    x1 =  K*math.exp(c)*x0 / (K + (math.exp(c) - 1)*x0)
    sys1 = VABSystemLogistic(K,r,x0)
    sys2 = VABSystemLogistic(K,r,x1)

    psensor = VABLogisticSensor([0,10**12])

    time.sleep(0.02)

    p1 = psensor.read(sys1)
    p2 = psensor.read(sys2)
    
    p1_trans =  K*math.exp(c)*p1 / (K + (math.exp(c) - 1)*p1)

    assert (p1_trans - 0.1) < p2 and p2 < (p1_trans + 0.1)


def test_VABSystemLogisticVirtual():
    K = 100
    c = 1
    r = 2
    x0 = 1
    x1 =  K*math.exp(c)*x0 / (K + (math.exp(c) - 1)*x0)
    sys1 = VABSystemLogisticVirtual(K,r,x0)
    sys2 = VABSystemLogisticVirtual(K,r,x1)

    psensor = VABLogisticSensorVirtual([0,10**12])
    tsensor = VABTimeSensor([0,10**12])
    pact = VABLogisticActuator_X([1,1000])
    tact = VABLogisticActuator_T([0,10**12])
    tact.set(sys1,0.02)
    tact.set(sys2,0.02)

    p1 = psensor.read(sys1)
    p2 = psensor.read(sys2)
    
    p1_trans =  K*math.exp(c)*p1 / (K + (math.exp(c) - 1)*p1)

    assert (p1_trans - 0.1) < p2 and p2 < (p1_trans + 0.1)


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


def test_Expression():
    exp1 = Expression("v[0]",set([0]),set([]))
    exp2 = Expression("c[0]",set([]),set([0]))
    assert exp1._left == None and exp1._right == None and exp1.CountParams() == 0
    assert exp2._left == None and exp2._right == None and exp2.CountParams() == 1
    assert exp1.Evaluate() == "v[0]"
    assert exp2.Evaluate() == "c[0]"
    assert exp1.Size() == 1 and exp2.Size() == 1
    exp1.SetLeft(Expression("v[1]",set([1]),set([])))
    exp1.SetTerminal("+")
    exp1.SetRight(Expression("v[0]",set([0]),set([])))
    assert exp1.Size() == 3 
    assert exp1.Evaluate()=="(v[1]+v[0])"
    func = FunctionTree(exp1)
    func.Operate(exp1,'+',exp2)
    assert exp1.Evaluate() == "((v[1]+v[0])+c[0])"
    assert exp1.CountParams() == 1


def test_FunctionTree():
#    exp1 = Expression("",0,0)
#    exp2 = Expression("",0,0)
#    exp1.SetLeft(Expression("c[0]",0,1))
#    exp1.SetTerminal("+")
#    exp2.SetLeft(Expression("c[1]",0,1))
#    exp2.SetTerminal("*")
#    exp2.SetRight(Expression("v[0]",1,0))
#    exp1.SetRight(exp2)
    pass
