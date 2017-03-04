# LotkaVolterraDemo.py

import eugene as eu
from eugene.src.virtual_sys.LotkaVolterra2D import *
import matplotlib.pyplot as plt


def LotkaVolterraDemo(noise_stdev=2., epsilon=10**(-3), resolution=[300,2], alpha=1):
    
    # build sensors and actuators
    tsensor = eu.sensors.VABTimeSensor([])
    x1sensor = eu.sensors.LotkaVolterra2DSensor(1, [-10**23,10.**23], noise_stdev, False)
    x2sensor = eu.sensors.LotkaVolterra2DSensor(2, [-10**23,10.**23], noise_stdev, False)
    x1act = eu.actuators.LotkaVolterra2DActuator(1, [-10**23.,10.**23])
    x2act = eu.actuators.LotkaVolterra2DActuator(2, [-10**23.,10.**23])
    tact = eu.actuators.VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(0, tsensor), (1, x1sensor), (2, x2sensor)])
    actuators = dict([(0, tact),(1, x1act), (2, x2act)])

    # build systems from data
    systems = []

    systems.append(LotkaVolterra2D(2., 4., 100., 100., 1., -1., 5., 5.))
    systems.append(LotkaVolterra2D(3., 3., 100., 100., 1., -1., 5., 5.))
    systems.append(LotkaVolterra2D(1.5, 3., 100., 100., 1., -1., 5., 5.))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(eu.interface.VABSystemInterface(sensors, actuators, sys))

    # build ROIs
    ROI = dict([(0, [0., 1.]), (1, [1., 100.]), (2, [1., 100.])])
            
    # collect data
    data = []
    for count, interface in enumerate(interfaces):
        print("Sampling data for system {}. ROI for time: {}. ROI for x1: {}.  ROI for x2: {}.\n".format(count, ROI[0], ROI[1], ROI[2]))
        data.append(eu.interface.TimeSampleData(0, [1,2], interface,
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
    f.show()

    # build models of the data
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(eu.compare.BuildSymModel(data_frame, 0, [1,2], sys_id,
            epsilon))

    # compare models
    comparison_pos1 = eu.compare.CompareModels(models[0], models[1])
    comparison_neg = eu.compare.CompareModels(models[0], models[2])
    comparison_pos2 = eu.compare.CompareModels(models[1], models[2])

    # report results
    print('Result of first positive comparison: {}'.format(comparison_pos1))
    print('Result of negative comparison: {}'.format(comparison_neg))
    print('Result of second positive comparison: {}'.format(comparison_pos2))
