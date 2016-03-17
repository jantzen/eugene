# CircuitDemo.py

import pdb
import eugene as eu
from eugene.src.virtual_sys.chaotic_circuits import *

def BuildModel(data_frame, sys_id, epsilon=0):
    model = eu.compare.BuildSymModel(data_frame, 3, [0,1,2], sys_id, epsilon)

    return model


def CircuitDemo(noise_stdev=.1, epsilon=10**(-4),
        resolution=[600,2],alpha=1):
    
    import matplotlib.pyplot as plt


    # build sensors and actuators
    tsensor = eu.sensors.VABTimeSensor([])
    xsensor0 = eu.sensors.CCVoltageSensor([-10**23,10.**23], 0, noise_stdev, False)
    xsensor1 = eu.sensors.CCVoltageSensor([-10**23,10.**23], 1, noise_stdev, False)
    xsensor2 = eu.sensors.CCVoltageSensor([-10**23,10.**23], 2, noise_stdev, False)
    xact0 = eu.actuators.CCVoltageActuator([0.,10.**23], 0)
    xact1 = eu.actuators.CCVoltageActuator([0.,10.**23], 0)
    xact2 = eu.actuators.CCVoltageActuator([0.,10.**23], 0)
    tact = eu.actuators.VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(3, tsensor), (0, xsensor0), (1, xsensor1), (2, xsensor2)])
    actuators = dict([(3,tact),(0,xact0), (1,xact1), (2,xact2)])

    # build systems from data
    systems = []

    systems.append(ChaoticCircuit(5))
    systems.append(ChaoticCircuit(6))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(eu.interface.VABSystemInterface(sensors, actuators, sys))

    # build ROIs
    ROI = []
    ROI.append(dict([(3,[0., 10.]), (0, np.array([-5.,5.])), (1,
        np.array([-5.,5])), (2, np.array([-5.,5.]))]))
    ROI.append(dict([(3,[0.,10.]), (0, np.array([-5.,5])), (1,
        np.array([-5.,5])), (2, np.array([-5.,5]))]))

            
    # collect data
    data = []
    for count, interface in enumerate(interfaces):
        print "Sampling data for system {}. ROI for time: {}. ROI for voltage, 1st deriv of voltage, 2nd deriv of voltage: {}.\n".format(count,
                ROI[count][3], ROI[count][0], ROI[count][1], ROI[count][2])
        data.append(eu.interface.TimeSampleData(3, [0,1,2], interface,
            ROI[count], resolution))

#    pdb.set_trace()

    # plot the raw data 
    f, ax = plt.subplots(2,2,sharey=True)
    ax = ax.flatten()
    f.set_size_inches(12,8)
    for sys in range(len(systems)):
        t = data[sys]._index_values
        x0 = (data[sys]._target_values[0])
        x1 = (data[sys]._target_values[1])
        current_axes = ax[sys]
        current_axes.plot(t, x0, '.')
        current_axes.set_xlabel('time')
        current_axes = ax[sys+2]
        current_axes.plot(t, x1, '.')
        current_axes.set_xlabel('time')
        # annotate
#        current_axes.text(annot[sys][1], annot[sys][2], annot[sys][0],
#                fontsize=12)
#        if sys == 0:
#            current_axes.set_ylabel('voltage')
    f.savefig('./outputs/circuit_fig1.png', dpi=300)

    # build models of the data
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(BuildModel(data_frame, sys_id, epsilon))


    comparison = eu.compare.CompareModels(models[0], models[1])

    print 'Result of comparison: {}'.format(comparison)
