# StatisticalMethodTesting.py

from VABProcess import *
from VABClasses import *

def SampleExponentialData():
    # set up a system, sensors, and actuators
    sys = VABSystemExpGrowth(1,0.2)
    tsensor = VABTimeSensor([])
    psensor = VABPopulationSensor([0,10**6])
    pact = VABPopulationActuator([0,10**6])
    
    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,psensor)])
    actuators = dict([(2,pact)])
    
    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)
    
    # build ROI
    ROI = dict([(1, [0,1]),(2,[1,100])])
    
    data_frame = TimeSampleData(1, 2, interface, ROI)

    return data_frame

def SampleLogisticData():
    # set up a system, sensors, and actuators
    sys = VABSystemLogistic(100,8,1)
    tsensor = VABTimeSensor([])
    psensor = VABLogisticSensor([0,10**6])
    pact = VABLogisticActuator([0,10**6])
    
    # build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1,tsensor),(2,psensor)])
    actuators = dict([(2,pact)])
    
    # build an interface
    interface = VABSystemInterface(sensors, actuators, sys)
    
    # build ROI
    ROI = dict([(1, [0,1]),(2,[1,100])])
    
    data_frame = TimeSampleData(1, 2, interface, ROI)

    return data_frame


def SampleReactionData():
    # set up systems, sensors, and actuators
    sys1 = VABSystemFirstOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(1,1)
    sys3 = VABSystemThirdOrderReaction(1,1)
    tsensor = VABTimeSensor([])
    xsensor = VABConcentrationSensor([0,3])
    xact = VABConcentrationActuator([0,1])

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(2,xact)])
    
    # build interfaces
    interface1 = VABSystemInterface(sensors, actuators, sys1)
    interface2 = VABSystemInterface(sensors, actuators, sys2)
    interface3 = VABSystemInterface(sensors, actuators, sys3)

    # build ROI
    ROI = dict([(1, [0,1]),(2,[0.1,1])])
    
    data_frames = [TimeSampleData(1, 2, interface1, ROI), TimeSampleData(1, 2, interface2, ROI), TimeSampleData(1, 2, interface3, ROI)]

    return data_frames

   


def BuildModel(data_frame, alpha=0):
    model = BuildSymModel(data_frame, 1, 2, alpha)

    return model