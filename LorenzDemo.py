# LorenzDemo.py

import eugene as eu
from eugene.src.virtual_sys.lorenz import *
import matplotlib.pyplot as plt


def LorenzDemo(noise_stdev=5., epsilon=10**(-4), resolution=[600,2], alpha=1):
    
    # build sensors and actuators
    tsensor = eu.sensors.VABTimeSensor([])
    xsensor = eu.sensors.LorenzSensor('x', [-10**23,10.**23], noise_stdev, False)
    ysensor = eu.sensors.LorenzSensor('y', [-10**23,10.**23], noise_stdev, False)
    zsensor = eu.sensors.LorenzSensor('z', [-10**23,10.**23], noise_stdev, False)
    xact = eu.actuators.LorenzActuator('x', [-10**23.,10.**23])
    yact = eu.actuators.LorenzActuator('y', [-10**23.,10.**23])
    zact = eu.actuators.LorenzActuator('z', [-10**23.,10.**23])
    tact = eu.actuators.VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(0, tsensor), (1, xsensor), (2, ysensor), (3, zsensor)])
    actuators = dict([(0, tact),(1, xact), (2, yact), (3, zact)])

    # build systems from data
    systems = []

    systems.append(LorenzSystem(8. / 3., 10., 28., 1., 1., 1.))
    systems.append(LorenzSystem(8. / 3., 10., 10., 1., 1., 1.))
    systems.append(LorenzSystem(8. / 3., 50., 28., 1., 1., 1.)),
    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(eu.interface.VABSystemInterface(sensors, actuators, sys))

    # build ROIs
    ROI = dict([(0, [0., 1.]), (1, [1., 2.]), (2, [1., 2.]), (3, [1., 2.])])
            
    # collect data
    data = []
    for count, interface in enumerate(interfaces):
        print "Sampling data for system {}. ROI for time: {}. ROI for x: {}. ROI for y: {}, ROI for z: {}.\n".format(count, ROI[0], ROI[1], ROI[2], ROI[3])
        data.append(eu.interface.TimeSampleData(0, [1,2,3], interface,
            ROI, resolution, True))


    # plot the raw data 
    f, ax = plt.subplots(2,3,sharey=True)
    ax = ax.flatten()
    f.set_size_inches(12,8)
    for sys in range(len(systems)):
        t = data[sys]._index_values
        x0 = (data[sys]._target_values[0])
        x1 = (data[sys]._target_values[1])
        current_axes = ax[sys]
        current_axes.plot(t, x0, '.')
        current_axes.set_xlabel('time')
        current_axes = ax[sys+3]
        current_axes.plot(t, x1, '.')
        current_axes.set_xlabel('time')
    f.savefig('./outputs/lorenz_raw.png', dpi=300)

    # build models of the data
    models = []
    for sys_id, data_frame in enumerate(data):
#        models.append(BuildModel(data_frame, sys_id, epsilon))
        models.append(eu.compare.BuildSymModel(data_frame, 0, [1,2,3], sys_id,
            epsilon))

    comparison_pos = eu.compare.CompareModels(models[0], models[1])
    comparison_neg = eu.compare.CompareModels(models[0], models[2])

    print 'Result of positive comparison: {}'.format(comparison_pos)
    print 'Result of negative comparison: {}'.format(comparison_neg)
