# TestVABClasses.py

from VABSystemExpGrowth import *

def test_sensors():
    sys1 = VABSystemExpGrowth(1, 0.02)
    sys2 = VABSystemExpGrowth(5, 0.02)

    tsensor1 = VABTimeSensor()
    tsensor2 = VABTimeSensor()

    psensor1 = VABPopulationSensor()
    psensor2 = VABPopulationSensor()

    pact1 = VABPopulationActuator()
    pact2 = VABPopulationActuator()

