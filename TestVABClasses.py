# TestVABClasses.py

from VABClasses import *

def test_sensors():
    sys = VABSystemExpGrowth(1, 0.2)
    tsensor = VABTimeSensor()
    t = tsensor.read(sys)
    ttest = t == sys._time
    psensor = VABPopulationSensor()
    p = psensor.read(sys)
    ptest = p == sys._population

    assert ttest and ptest

 
def test_actuators():
    sys = VABSystemExpGrowth(1, 0.2)
    pact = VABPopulationActuator()
    pact.set(sys,25)
    
    assert 25 == sys._population
  
   
   
def test_growth_model():   
    sys1 = VABSystemExpGrowth(1, 10)
    sys2 = VABSystemExpGrowth(5, 10)

    psensor = VABPopulationSensor()

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
    func.SetConstants([1,1])

    assert func.EvaluateAt([1]) == 2


def test_IncrementConstants():
    """The purpose of the function being tested is to change the constant indices
    in one function so that they do not comflict with those
    in another with which it is being combined.
    """
    func = Function("c[0]+c[1]+c[2]",3,0)
    func.IncrementConstants(3)

    assert func._const_count == 3 
    assert func._function == "c[3]+c[4]+c[5]"



def test_Function_Multiply():
    pass
