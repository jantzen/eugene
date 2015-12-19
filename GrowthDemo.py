# GrowthDemo

import eugene as eu
from eugene.src.virtual_sys.growth_sys import *

def BuildModel(data_frame, sys_id, epsilon=0):
    model = eu.compare.BuildSymModel(data_frame, 1, 2, sys_id, epsilon)

    return model


def GrowthDemo(noise_stdev=5., epsilon=10**(-4),
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

    # build models of the data
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(BuildModel(data_frame, sys_id, epsilon))

    # classify the systems
    classes = eu.categorize.Classify(range(len(systems)), models)

#    # build text for annotating plots
#    annot = [[]] * 2
#    annot[0] = [r'', 3*10**(-6), 0.8*10**(-4)]
#    annot[1] = [r'$HOI + O_3 \rightarrow IO_3^- $', 0.1, 0.8*10**(-4)]
    
    # plot the data (without indicating classification
    f, ax = plt.subplots(1,2,sharey=True)
    ax = ax.flatten()
    f.set_size_inches(12,8)
    for i, c in enumerate(classes):
        for sys in c._systems:
            t = data[sys]._index_values
            x = data[sys]._target_values
            x = np.array(x).transpose()
            current_axes = ax[sys]
            current_axes.plot(t, x, 'bo')
            current_axes.set_xlabel('time')
            # annotate
#            current_axes.text(annot[sys][1], annot[sys][2], annot[sys][0],
#                    fontsize=12)
            if sys == 0:
                current_axes.set_ylabel('population')
    f.savefig('./outputs/pop_growth_fig1.png', dpi=300)

    # replot the data (classified)
    colors = ['bo','go','ro','co']
#    plt.figure(2)
    f, ax = plt.subplots(1,2,sharey=True)
    ax = ax.flatten()
    f.set_size_inches(12,8)
    for i, c in enumerate(classes):
        for sys in c._systems:
#            plt.subplot(2,3,sys)
            t = data[sys]._index_values
            x = data[sys]._target_values
            x = np.array(x).transpose()
            current_axes = ax[sys]
            current_axes.plot(t, x, colors[i])
            current_axes.set_xlabel('time')
            # annotate
            # build text for annotating plots
            annot = [[]] * 2
            annot[0] = ['r = 0.5, K = 65, alpha = 1,\n beta = 0.6, gamma = 1.8',
                    5, 20]
            annot[1] = ['r = 1, K = 65, alpha = 0.5,\n beta = 1.5, gamma = 1.8',
                    5, 20]
            current_axes.text(annot[sys][1], annot[sys][2], annot[sys][0],
                    fontsize=12)
            if sys == 0:
                current_axes.set_ylabel('population')
    f.savefig('./outputs/pop_growth_fig2.png', dpi=300)

    return classes


