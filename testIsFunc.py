import numpy as np
import matplotlib.pyplot as plt
import math

from isFunction import isFunc

########################################################################
########################################################################
#create function data sets
####from Collin's "TestVABRangeDetermination.py (12/25/15)
           #do we want to maybe put this function (and other 
           #sorts of things like it) in a utilities folder/file/package?
def funcFillArray(function, length, delta=1, start=0):
    res = np.zeros(length)
    x = start
    for i in range(0, length):
        res[i] = function(x)
        x += delta
    return res
####

def getDataSets():
    delta=0.20
    length = 50
    ##################################
    ##################################
    #equations, & target arrays

    log1 = lambda x: 4 *  math.log(x+1)
    log1Arr = funcFillArray(log1, length, delta)

    #sin function
    sine = funcFillArray(math.sin, length, delta)


    parabola = lambda x: math.pow(x, 2)
    parab = funcFillArray(parabola, length, delta)
    parabola2 = lambda x: -math.pow(x, 2) + 4 * x
    parab2 = funcFillArray(parabola2, length, delta)
    parabola3 = lambda x: (x+10)*(x-3)*(x+3)*(x-7.5)
    parab3 = funcFillArray(parabola3, length, delta)    
    parabola4 = lambda x: ((x-5)**2 - 4)**2
    parab4 = funcFillArray(parabola4, length, delta)    
    ##################################
    ##################################
    #function arrays - delcarations
    f1 = np.empty([length, 2])
    f2 = np.empty([length, 2])
    f3 = np.empty([length, 2])
    f4 = np.empty([length, 2])
    f5 = np.empty([length, 2])
    f6 = np.empty([length, 2])
    #not function arrays
    nf1 = np.empty([length, 2])
    nf2 = np.empty([length, 2])
    nf3 = np.empty([length, 2])

    #fill arrays
    count = 0.
    for i in range(length):
        f1[i] = [count, log1Arr[i]]
        f2[i] = [count, sine[i]]
        f3[i] = [count, parab[i]]
        f4[i] = [count, parab2[i]]
        f5[i] = [count, parab3[i]]
        f6[i] = [count, parab4[i]]

        nf1[i] = [sine[i]*10, count] #this is not a function!
        nf2[i] = [parab3[i], count]
        nf3[i] = [parab4[i], count]

        count += delta


    return [f1, f2, f3, f4, f5, f6, nf1, nf2, nf3]


#def getNoisyDataSets()

########################################################################
########################################################################
########################################################################


#get list of datasets, some are funcs and other aren't
dss = getDataSets()

# for ds in dss:
#     isFunc(ds)
#     print

assert True, isFunc(dss[0])
assert True, isFunc(dss[1])
assert True, isFunc(dss[2])
assert True, isFunc(dss[3])
assert True, isFunc(dss[4])
assert True, isFunc(dss[5])
print "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNN\n"
assert False, isFunc(dss[6])
assert False, isFunc(dss[7])
assert False, isFunc(dss[8])

isFunc(dss[0])
isFunc(dss[6])
