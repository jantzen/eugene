# TestCompare.py

# from VABProcess import *
# from VABClasses import *
import eugene as eu
from eugene.virtual_sys.chemical_sys import *


def SampleReactionData(noise_stdev=0):
    # set up systems, sensors, and actuators
    sys1 = VABSystemFirstOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(1,1)
    sys3 = VABSystemThirdOrderReaction(1,1)
    sys4 = VABSystemSecondOrderReaction(0.5,2)
    tsensor = eu.connect.sensors.VABTimeSensor([])
    xsensor = eu.connect.sensors.VABConcentrationSensor([0,3], noise_stdev)
    xact = eu.connect.actuators.VABConcentrationActuator([0,1])

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(2,xact)])
    
    # build interfaces
    interface1 = eu.interface.VABSystemInterface(sensors, actuators, sys1)
    interface2 = eu.interface.VABSystemInterface(sensors, actuators, sys2)
    interface3 = eu.interface.VABSystemInterface(sensors, actuators, sys3)
    interface4 = eu.interface.VABSystemInterface(sensors, actuators, sys4)

    # build ROI
    ROI = dict([(1, [0,1]),(2,[0.1,1])])
    
    data_frames = [eu.interface.TimeSampleData(1, 2, interface1, ROI), eu.interface.TimeSampleData(1, 2,
        interface2, ROI), eu.interface.TimeSampleData(1, 2, interface3, ROI),
        eu.interface.TimeSampleData(1, 2, interface4, ROI)]

    return data_frames


def SampleSecondOrderReactionData(noise_stdev=0):
    # set up systems, sensors, and actuators
    sys1 = VABSystemSecondOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(0.1,1)
    sys3 = VABSystemSecondOrderReaction(1,0.1)
    sys4 = VABSystemSecondOrderReaction(0.5,2)
    tsensor = eu.connect.sensors.VABTimeSensor([])
    xsensor = eu.connect.sensors.VABConcentrationSensor([0,3], noise_stdev)
    xact = eu.connect.actuators.VABConcentrationActuator([0,1])
    tact = eu.connect.actuators.VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(1,tact),(2,xact)])
    
    # build interfaces
    interface1 = eu.interface.VABSystemInterface(sensors, actuators, sys1)
    interface2 = eu.interface.VABSystemInterface(sensors, actuators, sys2)
    interface3 = eu.interface.VABSystemInterface(sensors, actuators, sys3)
    interface4 = eu.interface.VABSystemInterface(sensors, actuators, sys4)

    # build ROI
    ROI = dict([(1, [0,1]),(2,[0.1,1])])
    
    data_frames = [eu.interface.TimeSampleData(1, 2, interface1, ROI), eu.interface.TimeSampleData(1, 2,
        interface2, ROI), eu.interface.TimeSampleData(1, 2, interface3, ROI),
        eu.interface.TimeSampleData(1, 2, interface4, ROI)]

    return data_frames


   
def BuildModel(data_frame, sys_id, epsilon=0):
    model = eu.compare.BuildSymModel(data_frame, 1, 2, sys_id, epsilon)

    return model



def testCompareModels(noise_stdev=0.001, proportional=False, epsilon=10**(-4),
        delta=10**(-3)):
    # set up systems, sensors, and actuators
    sys1 = VABSystemFirstOrderReaction(1,1)
    sys2 = VABSystemSecondOrderReaction(1,1)
    sys3 = VABSystemThirdOrderReaction(1,1)
    sys4 = VABSystemFirstOrderReaction(1,5)
    sys5 = VABSystemSecondOrderReaction(1,5)
    sys6 = VABSystemThirdOrderReaction(1,5)
    tsensor = eu.connect.sensors.VABTimeSensor([])
    xsensor = eu.connect.sensors.VABConcentrationSensor([-1,3], noise_stdev, proportional)
    xact = eu.connect.actuators.VABConcentrationActuator([0,1])
    tact = eu.connect.actuators.VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(1,tact),(2,xact)])
    
    # build interfaces
    interface1 = eu.interface.VABSystemInterface(sensors, actuators, sys1)
    interface2 = eu.interface.VABSystemInterface(sensors, actuators, sys2)
    interface3 = eu.interface.VABSystemInterface(sensors, actuators, sys3)
    interface4 = eu.interface.VABSystemInterface(sensors, actuators, sys4)
    interface5 = eu.interface.VABSystemInterface(sensors, actuators, sys5)
    interface6 = eu.interface.VABSystemInterface(sensors, actuators, sys6)

    # build ROIs
    ROI1 = dict([(1, [0.,np.log(2.)]),(2,[0.1,1.])])
    ROI2 = dict([(1, [0.,1./1.]),(2,[0.1,1])])
    ROI3 = dict([(1, [0.,3. / (2.*1.**2)]),(2,[0.1,1])])
    ROI4 = dict([(1, [0.,np.log(2.)/5.]),(2,[0.1,1])])
    ROI5 = dict([(1, [0.,1./(5. * 1.)]),(2,[0.1,1])])
    ROI6 = dict([(1, [0., 3./(10.*1.**2)]),(2,[0.1,1])])

    # get two sets of data for sys1
    [data11, data12] = [eu.interface.TimeSampleData(1,2,interface1,ROI1),
            eu.interface.TimeSampleData(1,2,interface1,ROI1)]

    # get a set of data for sys2 and sys3
    data21 = eu.interface.TimeSampleData(1,2,interface2,ROI2)
    data22 = eu.interface.TimeSampleData(1,2,interface2,ROI2)
    data31 = eu.interface.TimeSampleData(1,2,interface3,ROI3)
    data32 = eu.interface.TimeSampleData(1,2,interface3,ROI3)
    
    # get data for systems 4 through 6
    data41 = eu.interface.TimeSampleData(1,2,interface4,ROI4)
    data51 = eu.interface.TimeSampleData(1,2,interface5,ROI5)
    data61 = eu.interface.TimeSampleData(1,2,interface6,ROI6)

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
    out = [eu.compare.CompareModels(model11, model12, delta), eu.compare.CompareModels(model21, model22, delta), eu.compare.CompareModels(model31, model32, delta), eu.compare.CompareModels(model11, model21, delta), eu.compare.CompareModels(model11, model31, delta), eu.compare.CompareModels(model21, model31, delta),
            eu.compare.CompareModels(model11, model41, delta), eu.compare.CompareModels(model21, model51, delta), eu.compare.CompareModels(model31, model61, delta), eu.compare.CompareModels(model11, model51, delta), eu.compare.CompareModels(model11, model61, delta), eu.compare.CompareModels(model21, model61, delta)]

    print "Expected pattern: y, y, y, n, n, n, y, y, y, n, n, n\n"
    return out
