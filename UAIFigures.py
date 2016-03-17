# UAIFigures.py

""" Script for generating figures for the paper submitted to UAI 2016.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# Growth

# Generate the 2 x 2 plot describing the results of the growth experiment

# unpickle the confusion matrix for the noise and model experiments
f_noise = open('./outputs/CM_noise.pkl','rb')
f_model = open('./outputs/CM_model.pkl','rb')

CM_noise = pickle.load(f_noise)
CM_model = pickle.load(f_model)

f_noise.close()
f_model.close()

# Initialize the figure
fig_G = plt.figure(1)


# Compute the accuracy for each different noise level and add
# to the plot
xvals = []
yvals = []
for noise in CM_noise.keys():
    cm = np.array(CM_noise[noise],dtype='float64')
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm) * 100.
    x = noise
    y = accuracy
#    plt.annotate('FP = {}'.format(cm[0,1]),(x,y))
    xvals.append(x)
    yvals.append(y)
   
plt.subplot(2,1,1)
plt.plot(xvals, yvals, 'k+', markersize=12., markeredgewidth=2.5)

plt.xlabel('$\sigma$', fontsize=16, verticalalignment='center')
plt.ylabel('Accuracy (\%)', fontsize=14)
plt.ylim(30,105)
plt.xlim(0,33)
plt.title('(a)')

# Compute the accuracy for each different k value and add
# to the plot
xvals = []
yvals = []
for k in CM_model.keys():
    cm = np.array(CM_model[k],dtype='float64')
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm) * 100.
    x = 100. - k
    y = accuracy
#    plt.annotate('FP = {}'.format(cm[0,1]),(x,y))
#    plt.plot(x, y, 'k+', markersize=12.)
    xvals.append(x)
    yvals.append(y)

plt.subplot(2,1,2)
plt.plot(xvals, yvals, 'k+', markersize=12., markeredgewidth=2.5)

plt.xlabel('$k_1 - k_2$', fontsize=16, verticalalignment='center')
plt.ylabel('Accuracy (\%)', fontsize=14)
plt.ylim(30,105)
plt.xlim(5,0)
plt.title('(b)')

plt.show()

