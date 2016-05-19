# NIPSFigures.py

""" Script for generating figures for the paper submitted to NIPS 2016.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import eugene
#import pdb

###############################################################################
# Noise and skew

# Generate a 2x2 plot showing accuracy curves for both models for noise and skew
# experiments

# unpickle the confusion matrix for the logistic growth noise and skew
# experiments
f_LG_noise = open('./outputs/NIPS2016/LG_CM_noise.pkl','rb')
f_LG_skew = open('./outputs/NIPS2016/LG_CM_skew.pkl','rb')

LG_CM_noise = pickle.load(f_LG_noise)
LG_CM_skew = pickle.load(f_LG_skew)

f_LG_noise.close()
f_LG_skew.close()

# unpickle the confusion matrix for the Lotka-Volterra noise and skew
# experiments
f_LV_noise = open('./outputs/NIPS2016/LV_CM_noise.pkl','rb')
f_LV_skew = open('./outputs/NIPS2016/LV_CM_skew.pkl','rb')

LV_CM_noise = pickle.load(f_LV_noise)
LV_CM_skew = pickle.load(f_LV_skew)

f_LV_noise.close()
f_LV_skew.close()

# initialize the figure
fig_NS, ax_NS = plt.subplots(2,2,sharey=True)
plt.subplots_adjust(wspace=0.1,hspace=0.4,bottom=0.3)
ax_NS = ax_NS.flatten()

# compute the accuracy for each different noise level and add to plot (LG)
xvals = []
yvals = []
for noise in LG_CM_noise.keys():
    cm = np.array(LG_CM_noise[noise],dtype='float64')
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm) * 100.
    x = noise
    y = accuracy
    xvals.append(x)
    yvals.append(y)

current_axes = ax_NS[0]
current_axes.plot(xvals, yvals, 'k+', markersize=12., markeredgewidth=2.5)

current_axes.set_xlabel('$\sigma$', fontsize=16, verticalalignment='center',
        x=0.9)
current_axes.set_ylabel('Accuracy (\%)', fontsize=14)
current_axes.set_ylim(30,105)
current_axes.set_xlim(0,33)
current_axes.set_title('(a)', weight='bold')
#plt.text(28,90,'(a)',fontsize=16,weight='bold')

# compute the accuracy for each different skew level and add to plot (LG)
xvals = []
yvals = []
for skew in LG_CM_skew.keys():
    cm = np.array(LG_CM_skew[skew],dtype='float64')
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm) * 100.
    x = skew
    y = accuracy
    xvals.append(x)
    yvals.append(y)
   
current_axes = ax_NS[1]
current_axes.plot(xvals, yvals, 'k+', markersize=12., markeredgewidth=2.5)

current_axes.set_xlabel('$\\alpha$', fontsize=16, verticalalignment='center',
        x=0.9)
current_axes.set_ylim(30,105)
current_axes.set_xlim(-0.5,33)
current_axes.set_title('(b)', weight='bold')
#plt.text(28,90,'(b)',fontsize=16,weight='bold')

# compute the accuracy for each different noise level and add to plot (LV)
xvals = []
yvals = []
for noise in LV_CM_noise.keys():
    cm = np.array(LV_CM_noise[noise],dtype='float64')
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm) * 100.
    x = noise
    y = accuracy
    xvals.append(x)
    yvals.append(y)
   
current_axes = ax_NS[2]
current_axes.plot(xvals, yvals, 'k+', markersize=12., markeredgewidth=2.5)

current_axes.set_xlabel('$\sigma$', fontsize=16, verticalalignment='center',
        x=0.9)
current_axes.set_ylabel('Accuracy (\%)', fontsize=14)
current_axes.set_ylim(30,105)
current_axes.set_xlim(0,33)
current_axes.set_title('(c)', weight='bold')
#plt.text(28,90,'(c)',fontsize=16,weight='bold')

# compute the accuracy for each different skew level and add to plot (LV)
xvals = []
yvals = []
for skew in LV_CM_skew.keys():
    cm = np.array(LV_CM_skew[skew],dtype='float64')
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm) * 100.
    x = skew
    y = accuracy
    xvals.append(x)
    yvals.append(y)
   
current_axes = ax_NS[3]
current_axes.plot(xvals, yvals, 'k+', markersize=12., markeredgewidth=2.5)

current_axes.set_xlabel('$\\alpha$', fontsize=16, verticalalignment='center',
        x=0.9)
current_axes.set_ylim(30,105)
current_axes.set_xlim(-0.5,33)
current_axes.set_title('(d)', weight='bold')
#plt.text(28,90,'(d)',fontsize=16,weight='bold')

###############################################################################
# Classification (Growth models)

# initialize the figure
f_GS, ax = plt.subplots(2,3)

### DIFFERENT KINDS

### local variables
epsilon=10**(-4)
resolution=[300,2]
noise_stdev = 5.
beta = 0.6
sp1 = [1.5, 1, 100, 1, 1, 1, 0]
sp2 = [1.5, 1, 100, 1, beta, 1, 0]
###

###
#index sensor & actuator
isensor = eugene.sensors.VABTimeSensor([])
iact = eugene.actuators.VABVirtualTimeActuator()

#target sensor & actuator
tsensor = eugene.sensors.PopulationSensor([-10**(23), 10**23], noise_stdev, 
                                      False)
tact = eugene.actuators.PopulationActuator([-10**23, 10**23])

sensors = dict([(1, isensor), (2, tsensor)])
actuators = dict([(1, iact), (2, tact)])
###

###
systems = []

systems.append(eugene.growth_sys.LogisticGrowthModel(sp1[0], sp1[1], sp1[2], sp1[3],
                                   sp1[4], sp1[5], sp1[6]))


systems.append(eugene.growth_sys.LogisticGrowthModel(sp2[0], sp2[1], sp2[2], sp2[3],
                                   sp2[4], sp2[5], sp2[6]))


interfaces = []
for sys in systems:
    interfaces.append(eugene.interface.VABSystemInterface(sensors, actuators, sys))

ROI = dict([(1,[0., 10.]), (2, [1., 51.])])
###    

### collect data!
data = []
for iface in interfaces:
    data.append(eugene.interface.TimeSampleData(1, [2], iface, ROI, resolution))
###

# plot the data 
colors = ['bo','go','ro','co']
ax = ax.flatten()
f_GS.set_size_inches(12,8)

for i, d in enumerate(data):
    t = d._index_values
    x = np.hstack(d._target_values)
    current_axes = ax[i]
    current_axes.plot(t, x[:,0], 'k+', t, x[:,1], 'k.')
    current_axes.set_xlabel('time', verticalalignment='center')
    # annotate
    # build text for annotating plots
    annot = [[]] * 2
    annot[0] = ['r = 1.5, K = 100, alpha = 1,\n beta = 1, gamma = 1',
            3, 3]
    annot[1] = ['r = 1.5, K = 100, alpha = 1,\n beta = 1, gamma = 1',
            3, 3]            
    current_axes.text(annot[i][1], annot[i][2], annot[i][0], fontsize=12)
    current_axes.set_ylabel('population')
 
### plot the accuracy curve
# unpickle the confusion matrix for the different kind comparison experiment
f_LG_different = open('./outputs/NIPS2016/LG_CM_different.pkl','rb')

LG_CM_different = pickle.load(f_LG_different)

f_LG_different.close()

# compute the accuracy for each different beta value and add to plot 
xvals = []
yvals = []
for beta in LG_CM_different.keys():
    cm = np.array(LG_CM_different[beta],dtype='float64')
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm) * 100.
    x = beta
    y = accuracy
    xvals.append(x)
    yvals.append(y)

current_axes = ax[2]
current_axes.plot(xvals, yvals, 'k+', markersize=12., markeredgewidth=2.5)

current_axes.set_xlabel('$\\beta$', fontsize=16, verticalalignment='center',
        x=0.9)
current_axes.set_ylabel('Accuracy (\%)', fontsize=14)
current_axes.set_ylim(0,105)
current_axes.set_xlim(-0.1,1.1)
current_axes.set_title('(c)')
#
plt.show()
