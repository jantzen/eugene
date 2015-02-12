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
    sys1 = VABSystemExpGrowth(1, 0.2)
    sys2 = VABSystemExpGrowth(5, 0.2)

    psensor = VABPopulationSensor()

    time.sleep(5)
 
    p1 = psensor.read(sys1)
    p2 = psensor.read(sys2)
    
    print(5*p1)
    print(p2)

    assert (5*p1 - .1) < p2 and p2 < (5*p1 + .1)
    
