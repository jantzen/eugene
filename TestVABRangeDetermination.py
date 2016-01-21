#TestVABRangeDetermination.py
#Determining a range is fairly subjective, so this really functions
#more as a trial, than a test, but this can be used to test the helper functions
#as well.

import VABRangeDetermination as V
import matplotlib.pyplot as plt
import numpy as np
from VABClasses import *
from VABProcess import *
import math

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
    systems.append(VABSystemFirstOrderReaction(10.**(-6), 1.1*10.**5))
    
    # second-order
    # HOI + O3
    systems.append(VABSystemSecondOrderReaction(10.**(-6),
            3.6*10.**4))
    # OI- + O3
    systems.append(VABSystemSecondOrderReaction(10.**(-6), 1.6*10.**6))
    # HOI + HOCl
    systems.append(VABSystemSecondOrderReaction(10.**(-6), 8.2))
    # HOI + OCl-
    systems.append(VABSystemSecondOrderReaction(10.**(-6), 52))

    # third-order
    # HOI + HOCl + HOCl
    systems.append(VABSystemThirdOrderReaction(10.**(-6),
            8.3*10.**4))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(VABSystemInterface(sensors, actuators, sys))

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
        print "Sampling data for system {}. ROI for time: {}. ROI for concentration: {}.\n".format(count, 
                ROIs[count][1], ROIs[count][2])
        data.append(TimeSampleData(1, 2, interface, ROIs[count]))
        
    return data

def funcFillArray(function, length, delta=1, start=0):
    res = np.zeros(length)
    x = start
    for i in range(0, length):
        res[i] = function(x)
        x += delta
    return res

inLog = lambda x: 1/math.log(x+1)

parabola = lambda x: math.pow(x, 2)

def bump(x):
    if abs(x) < 1:
        return  math.pow(math.e, -(1/(1-math.pow(x,2))))
    else:
        return 0

b = np.array([2,3,4,5,5.5,6,6.5,7,6.5,6,5.5,5,4,3,2,1.5,1,1,1.5,1])
a = np.array([2303, 210, 110, 75, 57, 46, 38, 33, 29, 26, 23, 21, 20, 18, 17, 16, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
c = funcFillArray(math.sin, 1000, .1)
d = funcFillArray(inLog, 50, .1, .1)
e = funcFillArray(parabola, 50, 1, -24)
f = funcFillArray(parabola, 50, 1, 50)
g = funcFillArray(bump, 50, .0625, -1.5)
#data = CLMPSdemo()[1]._target_values

#for array in data:
#    result = V.findRange(array)
#    plt.plot(array, "b")
#    stop = (int)(result.start+result.data.size)
#    start = (int)(result.start)
#    plt.plot(range(start, stop), result.data, "r")
#i=3

test = f

result = V.findRange(test)
plt.plot(test, "b")
stop = (int)(result.start+result.data.size)
start = (int)(result.start)

plt.plot(range(start, stop), result.data, "r")
#plt.plot(np.diff(test), "y")
#plt.plot(np.diff(test, n=2), "g")

#monotone = V.findMonotone(c)
#plt.plot(test[0:monotone], "y")
trans = V.curveFind2(test)
plt.plot(trans, "y")
plt.show()