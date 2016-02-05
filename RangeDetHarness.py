import eugene as eu
from eugene.src.virtual_sys.growth_sys import *
import VABRangeDetermination as V

def BuildModel(data_frame, sys_id, epsilon=0):
    model = eu.compare.BuildSymModel(data_frame, 1, 2, sys_id, epsilon)

    return model
    
def DemoWithRange(noise_stdev=5., epsilon=10**(-4),
        resolution=[300,3],alpha=1):
        
    import matplotlib.pyplot as plt
    
    # build sensors and actuators
    tsensor = eu.sensors.VABTimeSensor([])
    xsensor = eu.sensors.PopulationSensor([-10**23,10.**23], noise_stdev,
            False)
    xact = eu.actuators.PopulationActuator([0.,10.**23])
    tact = eu.actuators.VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(1,tact),(2,xact)])

    # build systems from data
    systems = []

    systems.append(LogisticGrowthModel(0.5, 5., 65., 1.5, 0.5, 1.8, 0.))
    systems.append(LogisticGrowthModel(1., 5., 65., 0.8, 1.5, 1.8, 0.))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(eu.interface.VABSystemInterface(sensors, actuators, sys))

    # build ROIs
    ROI = dict([(1,[0., 20.]), (2, [5., 65.])])
            
    # collect data
    data = []
    for count, interface in enumerate(interfaces):
        print "Sampling data for system {}. ROI for time: {}. ROI for concentration: {}.\n".format(count, ROI[1], ROI[2])
        data.append(eu.interface.TimeSampleData(1, 2, interface, ROI,
            resolution))
            
    #print data[0]._index_values
    #print data[0]._target_values[0]
    
    i = 0
    for experiment in data[0]._target_values:
        ranged = V.findRange(experiment)
        data[0]._target_values[i] = ranged
        i += 1
        
    # build models of the data
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(BuildModel(data_frame, sys_id, epsilon))

    # classify the systems
    classes = eu.categorize.Classify(range(len(systems)), models)
    
    

# Main Script
DemoWithRange()