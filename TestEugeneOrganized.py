#to be determined:
import eugene.TEMP_unorganized as unorg

#Eugene
from   eugene import interface as iface
from   eugene.connect.sensors import *
from   eugene.connect.actuators import *
from   eugene.virtual_sys import chemical_sys as cs
import eugene.categorize as catgz
import eugene.compare as compare

#regular python modules
import matplotlib.pyplot as plt

def CLMPSdemo(noise_stdev=0.01, epsilon=10**(-4)):
    


    # build sensors and actuators
    tsensor = VABTimeSensor([])
    xsensor = VABConcentrationSensor([0.,10.**23], noise_stdev, True)
    xact = VABConcentrationActuator([0.,10.**23])
    tact = VABVirtualTimeActuator()

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(1, tsensor), (2, xsensor)])
    actuators = dict([(1,tact),(2,xact)])

    # build systems from data
    systems = []

    # first-order
    # HO3 --> OH + O2 
    systems.append(cs.VABSystemFirstOrderReaction(10.**(-6), 1.1*10.**5))
    
    # second-order
    # HOI + O3
    systems.append(cs.VABSystemSecondOrderReaction(10.**(-6),
            3.6*10.**4))
    # OI- + O3
    systems.append(cs.VABSystemSecondOrderReaction(10.**(-6), 1.6*10.**6))
    # HOI + HOCl
    systems.append(cs.VABSystemSecondOrderReaction(10.**(-6), 8.2))
    # HOI + OCl-
    systems.append(cs.VABSystemSecondOrderReaction(10.**(-6), 52))

    # third-order
    # HOI + HOCl + HOCl
    systems.append(cs.VABSystemThirdOrderReaction(10.**(-6),
            8.3*10.**4))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(iface.VABSystemInterface(sensors, actuators, sys))

    # build ROIs
    ROIs = []
    for counter, sys in enumerate(systems):
        if counter == 0:
            ROIs.append(dict([(1, [0., np.log(2)/sys._k]),(2,
                [10.**(-6),10.**(-4)])]))
        elif counter == 1:
            ROIs.append(dict([(1, [0., 1./(sys._k *
                (10.**(-4)))]),(2,[10.**(-6),10.**(-4)])]))
        elif counter == 2:
            ROIs.append(dict([(1, [0., 1./(sys._k *
                (10.**(-4)))]),(2,[10.**(-6),10.**(-4)])]))
        elif counter == 3:
            ROIs.append(dict([(1, [0., 1./(sys._k * (10**(-4)))]),(2,[10.**(-6),10.**(-4)])]))
        elif counter == 4:
            ROIs.append(dict([(1, [0.,
                1./(sys._k*10.**(-4))]),(2,[10.**(-6),10.**(-4)])]))
        elif counter == 5:
            ROIs.append(dict([(1, [0.,
                3./(2.*sys._k*(10.**(-4))**2)]),(2,[10.**(-6),10.**(-4)])]))

            
    # collect data
    data = []
    for count, interface in enumerate(interfaces):
        print "Sampling data for system {}. ROI for time: {}. ROI for concentration: {}.\n".format(count, ROIs[count][1], ROIs[count][2])
        data.append(iface.TimeSampleData(1, 2, interface, ROIs[count]))

    # plot the data (raw)
#    plt.figure(1)
#    plt.subplots(2,3,sharey=True)
#    for i, frame in enumerate(data):
#        plt.subplot(2,3,i+1)
#        t = frame._index_values
#        x = frame._target_values
#        x = np.array(x).transpose()
#        plt.plot(t, x, 'bo')
#    plt.show

    # build models of the data
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(compare.BuildModel(data_frame, sys_id, epsilon))

    # classify the systems
    classes = catgz.Classify(range(len(systems)), models)

    # build text for annotating plots
    annot = [[]] * 6
    annot[0] = [r'$HO_3 \rightarrow OH + O_2$', 3*10**(-6), 0.8*10**(-4)]
    annot[1] = [r'$HOI + O_3 \rightarrow IO_3^- $', 0.1, 0.8*10**(-4)]
    annot[2] = [r'$OI^- + O_3 \rightarrow IO_3^- $', 3*10**(-3), 0.8*10**(-4)]
    annot[3] = [r'$HOI + HOCl \rightarrow IO_3^- $', 600, 0.8*10**(-4)]
    annot[4] = [r'$HOI + OCl \rightarrow IO_3^- $', 75, 0.8*10**(-4)]
    annot[5] = [r'$HOI + HOCl + HOCl \rightarrow IO_3^- $', 450, 0.8*10**(-4)]
    
    # plot the data (without indicating classification
    f, ax = plt.subplots(2,3,sharey=True)
    ax = ax.flatten()
    f.set_size_inches(12,8)
    for i, c in enumerate(classes):
        for sys in c._systems:
#            plt.subplot(2,3,sys)
            t = data[sys]._index_values
            x = data[sys]._target_values
            x = np.array(x).transpose()
            current_axes = ax[sys]
            current_axes.plot(t, x, 'bo')
            current_axes.set_xlabel('time [s]')
            # make axes display in exponential notation in desired range
            current_axes.get_yaxis().get_major_formatter().set_powerlimits((-3,
                3))
            current_axes.get_xaxis().get_major_formatter().set_powerlimits((-3,
                3))
            # annotate
            current_axes.text(annot[sys][1], annot[sys][2], annot[sys][0],
                    fontsize=12)
            if sys == 0:
                current_axes.set_ylabel('[X] [moles/L]')
    f.savefig('./outputs/fig1.png', dpi=300)

    # replot the data (classified)
    colors = ['bo','go','ro']
#    plt.figure(2)
    f, ax = plt.subplots(2,3,sharey=True)
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
            # make axes display in exponential notation in desired range
            current_axes.get_yaxis().get_major_formatter().set_powerlimits((-3,
                3))
            current_axes.get_xaxis().get_major_formatter().set_powerlimits((-3,
                3))           
            current_axes.set_xlabel('time [s]')
            # annotate
            current_axes.text(annot[sys][1], annot[sys][2], annot[sys][0],
                    fontsize=12)
            if sys == 0:
                current_axes.set_ylabel('[X] [moles/L]')
    f.savefig('./outputs/fig3.png', dpi=300)

    return classes

####################################################
CLMPSdemo()
