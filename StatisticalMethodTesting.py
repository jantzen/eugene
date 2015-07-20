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


def SampleReactionData(noise_stdev=0):
    # set up systems, sensors, and actuators
    sys1 = VABSystemFirstOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(1,1)
    sys3 = VABSystemThirdOrderReaction(1,1)
    sys4 = VABSystemSecondOrderReaction(0.5,2)
    tsensor = VABTimeSensor([])
    xsensor = VABConcentrationSensor([0,3], noise_stdev)
    xact = VABConcentrationActuator([0,1])

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(2,xact)])
    
    # build interfaces
    interface1 = VABSystemInterface(sensors, actuators, sys1)
    interface2 = VABSystemInterface(sensors, actuators, sys2)
    interface3 = VABSystemInterface(sensors, actuators, sys3)
    interface4 = VABSystemInterface(sensors, actuators, sys4)

    # build ROI
    ROI = dict([(1, [0,1]),(2,[0.1,1])])
    
    data_frames = [TimeSampleData(1, 2, interface1, ROI), TimeSampleData(1, 2,
        interface2, ROI), TimeSampleData(1, 2, interface3, ROI),
        TimeSampleData(1, 2, interface4, ROI)]

    return data_frames


def SampleSecondOrderReactionData(noise_stdev=0):
    # set up systems, sensors, and actuators
    sys1 = VABSystemSecondOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(0.1,1)
    sys3 = VABSystemSecondOrderReaction(1,0.1)
    sys4 = VABSystemSecondOrderReaction(0.5,2)
    tsensor = VABTimeSensor([])
    xsensor = VABConcentrationSensor([0,3], noise_stdev)
    xact = VABConcentrationActuator([0,1])
    tact = VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(1,tact),(2,xact)])
    
    # build interfaces
    interface1 = VABSystemInterface(sensors, actuators, sys1)
    interface2 = VABSystemInterface(sensors, actuators, sys2)
    interface3 = VABSystemInterface(sensors, actuators, sys3)
    interface4 = VABSystemInterface(sensors, actuators, sys4)

    # build ROI
    ROI = dict([(1, [0,1]),(2,[0.1,1])])
    
    data_frames = [TimeSampleData(1, 2, interface1, ROI), TimeSampleData(1, 2,
        interface2, ROI), TimeSampleData(1, 2, interface3, ROI),
        TimeSampleData(1, 2, interface4, ROI)]

    return data_frames


   
def BuildModel(data_frame, sys_id, epsilon=0):
    model = BuildSymModel(data_frame, 1, 2, sys_id, epsilon)

    return model


def testSymTestTemporal(num_trans=1, epsilon=0, noise_stdev=0):
    # get data
    # set up systems, sensors, and actuators
    sys1 = VABSystemFirstOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(1,1)
    sys3 = VABSystemThirdOrderReaction(1,1)
    tsensor = VABTimeSensor([])
    xsensor = VABConcentrationSensor([-1,3], noise_stdev)
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
    sys11 = VABSystemFirstOrderReaction(1,2,noise_stdev)
    sys12 = VABSystemFirstOrderReaction(0.5,1,noise_stdev)
    sys13 = VABSystemFirstOrderReaction(1,0.5,noise_stdev)

    interface11 = VABSystemInterface(sensors, actuators, sys11)
    interface12 = VABSystemInterface(sensors, actuators, sys12)
    interface13 = VABSystemInterface(sensors, actuators, sys13)

    assert SymTestTemporal(model1, interface11, 1, 2, ROI, num_trans)
    assert SymTestTemporal(model1, interface12, 1, 2, ROI, num_trans)
    assert SymTestTemporal(model1, interface13, 1, 2, ROI, num_trans)


def testCompareModels(noise_stdev=0.001, proportional=False, epsilon=10**(-4)):
    # set up systems, sensors, and actuators
    sys1 = VABSystemFirstOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(1,1)
    sys3 = VABSystemThirdOrderReaction(1,1)
    sys4 = VABSystemFirstOrderReaction(1,5)
    sys5 = VABSystemSecondOrderReaction(1,5)
    sys6 = VABSystemThirdOrderReaction(1,5)
    tsensor = VABTimeSensor([])
    xsensor = VABConcentrationSensor([-1,3], noise_stdev, proportional)
    xact = VABConcentrationActuator([0,1])
    tact = VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(1,tact),(2,xact)])
    
    # build interfaces
    interface1 = VABSystemInterface(sensors, actuators, sys1)
    interface2 = VABSystemInterface(sensors, actuators, sys2)
    interface3 = VABSystemInterface(sensors, actuators, sys3)
    interface4 = VABSystemInterface(sensors, actuators, sys4)
    interface5 = VABSystemInterface(sensors, actuators, sys5)
    interface6 = VABSystemInterface(sensors, actuators, sys6)

    # build ROIs
    ROI1 = dict([(1, [0.,np.log(2.)]),(2,[0.1,1.])])
    ROI2 = dict([(1, [0.,1./0.1]),(2,[0.1,1])])
    ROI3 = dict([(1, [0.,3. / (2.)]),(2,[0.1,1])])
    ROI4 = dict([(1, [0.,np.log(2.)/5.]),(2,[0.1,1])])
    ROI5 = dict([(1, [0.,1./(5. * 0.1)]),(2,[0.1,1])])
    ROI6 = dict([(1, [0., 3./(10.)]),(2,[0.1,1])])

    # get two sets of data for sys1
    [data11, data12] = [TimeSampleData(1,2,interface1,ROI1),
            TimeSampleData(1,2,interface1,ROI1)]

    # get a set of data for sys2 and sys3
    data21 = TimeSampleData(1,2,interface2,ROI2)
    data22 = TimeSampleData(1,2,interface2,ROI2)
    data31 = TimeSampleData(1,2,interface3,ROI3)
    data32 = TimeSampleData(1,2,interface3,ROI3)
    
    # get data for systems 4 through 6
    data41 = TimeSampleData(1,2,interface4,ROI4)
    data51 = TimeSampleData(1,2,interface5,ROI5)
    data61 = TimeSampleData(1,2,interface6,ROI6)

    # build models
    [model11, model12] = [BuildModel(data11, 1, epsilon), BuildModel(data12, 1,
        epsilon)]
    [model21, model22] = [BuildModel(data21, 2, epsilon), BuildModel(data22, 2,
        epsilon)]
    [model31, model32] = [BuildModel(data31, 3, epsilon), BuildModel(data32, 3,
        epsilon)]
    [model41, model51, model61] = [BuildModel(data41, 4, epsilon),
            BuildModel(data51, 5, epsilon), BuildModel(data61, 6, epsilon)]

    # compare models
    out = [CompareModels(model11, model12), CompareModels(model21, model22),
            CompareModels(model31, model32), CompareModels(model11, model21),
            CompareModels(model11, model31), CompareModels(model21, model31),
            CompareModels(model11, model41), CompareModels(model21, model51),
            CompareModels(model31, model61), CompareModels(model11, model51), 
            CompareModels(model11, model61), CompareModels(model21, model61)]

    print "Expected pattern: y, y, y, n, n, n, y, y, y, n, n, n\n"
    return out

def testClassifier(noise_stdev=0.001, epsilon=10**(-4)):
    # build sensors and actuators
    tsensor = VABTimeSensor([])
    xsensor = VABConcentrationSensor([-1,3], noise_stdev)
    xact = VABConcentrationActuator([0,1])

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(2,xact)])

    # build three of each type of system with randomly chosen parameters
    systems = []
    for i in range(3):
        systems.append(VABSystemFirstOrderReaction(np.random.uniform(0.1,1.0),
            np.random.uniform(0.5,1)))
    for i in range(3):
        systems.append(VABSystemSecondOrderReaction(np.random.uniform(0.1,1.0),
            np.random.uniform(0.5,1)))
    for i in range(3):
        systems.append(VABSystemThirdOrderReaction(np.random.uniform(0.1,1.0),
            np.random.uniform(0.5,1)))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(VABSystemInterface(sensors, actuators, sys))

    # build ROI
    ROI = dict([(1, [0,1]),(2,[0.1,1])])

    # collect data
    data = []
    for count, interface in enumerate(interfaces):
        print "Sampling data for system {}...\n".format(count)
        data.append(TimeSampleData(1, 2, interface, ROI))

    # build models of the data
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(BuildModel(data_frame, sys_id, epsilon))

    # classify the systems
    classes = Classify(range(len(systems)), models)

    return classes
