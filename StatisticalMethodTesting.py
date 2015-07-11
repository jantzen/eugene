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

   


def BuildModel(data_frame, epsilon=0):
    model = BuildSymModel(data_frame, 1, 2, epsilon)

    return model


def testSymTestTemporal(num_trans=1, epsilon=0):
    # get data
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
    
    [data1, data2, data3] = [TimeSampleData(1, 2, interface1, ROI), TimeSampleData(1, 2, interface2, ROI), TimeSampleData(1, 2, interface3, ROI)]

    # build models
    [model1, model2, model3] = [BuildModel(data1, epsilon), BuildModel(data2,
        epsilon), BuildModel(data3, epsilon)]
    
    # now check each model against the system for which it was built
    assert SymTestTemporal(model1, interface1, 1, 2, ROI, num_trans)
    assert SymTestTemporal(model2, interface2, 1, 2, ROI, num_trans)
    assert SymTestTemporal(model3, interface3, 1, 2, ROI, num_trans)


    # now check models against systems for which it was NOT built
    assert not SymTestTemporal(model1, interface2, 1, 2, ROI, num_trans)
    assert not SymTestTemporal(model1, interface3, 1, 2, ROI, num_trans)
    assert not SymTestTemporal(model2, interface3, 1, 2, ROI, num_trans)
    assert not SymTestTemporal(model3, interface1, 1, 2, ROI, num_trans)

    # finally, check and make sure that reactions of the same order end up in
    # the same category despite different reaction rates
    sys11 = VABSystemFirstOrderReaction(1,2)
    sys12 = VABSystemFirstOrderReaction(0.5,1)
    sys13 = VABSystemFirstOrderReaction(1,0.5)

    interface11 = VABSystemInterface(sensors, actuators, sys11)
    interface12 = VABSystemInterface(sensors, actuators, sys12)
    interface13 = VABSystemInterface(sensors, actuators, sys13)

    assert SymTestTemporal(model1, interface11, 1, 2, ROI, num_trans)
    assert SymTestTemporal(model1, interface12, 1, 2, ROI, num_trans)
    assert SymTestTemporal(model1, interface13, 1, 2, ROI, num_trans)

 


