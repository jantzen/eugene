import numpy as np
import matplotlib.pyplot as plt
import math

import eugene as eu
import eugene.src.virtual_sys.chemical_sys as cs
import isFunction as isF

########################################################################
########################################################################

# build abunch of data for the TestCase, which is at about line 130

def funcFillArray(function, length, delta=1, start=0):
    """
    create function data sets
    From Collin's TestVABRangeDetermination
    """
    res = np.zeros(length)
    x = start
    for i in range(0, length):
        res[i] = function(x)
        x += delta
    return res

def getMoreDataFrames(res=102, noise=10**(-3)):
    """
    provides getDataFrames with "more" data frames.
    "more" data frames includes:
        1 clean with flag & is function
        1 clean with flag & isn't function
        1 noisy with flag & is function
        1 noisy with flag & isn't function
    
    """

    delt=0.1
    DFs = []


    iv = np.ndarray([res])
    for i in range(res):
        iv[i] = i*delt

    #1 DF, clean w/ flag & is function (though trivial)
    tv = []
    for s in range(10):
        t = np.ndarray([res])
        t.fill(s)
        tv.append(t)
    DFs.append(eu.interface.DataFrame(1, iv, 2, tv))

    #1 noisy with flag & is function
    polys = [(lambda x: x**2), (lambda x: x**2 + 1), (lambda x: x**2 + 2), 
             (lambda x: x**2 + 3), (lambda x: x**2 + 4), 
             (lambda x: x**2 + 50), (lambda x: x**2 + 60), 
             (lambda x: x**2 + 70),  (lambda x: x**2 + 80), 
             (lambda x: x**2 + 90)]

    tvPoly = []
    for s in range(10):
        t = np.ndarray([res])
        t = funcFillArray(polys[s], res, delta=delt, start=-5)
        #put in some noise. --ex:sensors.Line52-64
        for i in range(len(t)):
            t[i] = t[i] + (np.random.normal(0, noise))
        tvPoly.append(t)
    DFs.append(eu.interface.DataFrame(1, iv, 2, tvPoly))
    
    
    return DFs
    

def getDataFrames():
    """
       Makes 3 data samples that is a function.
       Gets more data samples from getMoreDataFrames; some are functions,
       and others are not.

       return list of data samples.  
       -currently returns list of one DataSample, for simplicity.
    """

    #make ONE function data set
    tsensor = eu.sensors.VABTimeSensor([])
    csensor = eu.sensors.VABConcentrationSensor([0.,10.**23], 
                                                noise_stdev=10.**(-9))
                                                # noise_stdev=10.**(-6))
                                                #-> 'outofrange' values
    #if concentration > range[1] or concentration < range[0], then
         #return 'outofrange'
         #^ in eu.sensors->Line61



    sensors = dict([(1, tsensor), (2, csensor)])
    tact = eu.actuators.VABVirtualTimeActuator()
    cact = eu.actuators.VABConcentrationActuator([])
    actuators = dict([(1, tact), (2, cact)])
    sf1 = cs.VABSystemFirstOrderReaction(10.**(-6), 1.1*10.**5) 
    sf2 = cs.VABSystemFirstOrderReaction(10.**(-6), 2.1*10.**5) 
    sf3 = cs.VABSystemThirdOrderReaction(10.**(-6), 8.3*10.**4) 
    
    ifaces = []
    datas = []
    ifaces.append(eu.interface.VABSystemInterface(sensors, actuators, sf1))
    ifaces.append(eu.interface.VABSystemInterface(sensors, actuators, sf2))
    ifaces.append(eu.interface.VABSystemInterface(sensors, actuators, sf3))
    ROIs = [dict([(1, [0., np.log(2)/sf1._k]),
                  (2, [10.**(-6),10.**(-4)])]),
            dict([(1, [0., np.log(2)/sf2._k]),
                  (2, [10.**(-6),10.**(-4)])]),
            dict([(1, [0., 3./(2.*sf3._k*(10.**(-4))**2)]),
                  (2,[10.**(-6),10.**(-4)])])]

    for r, iface in enumerate(ifaces):
        datas.append(eu.interface.TimeSampleData(1,2, iface, ROIs[r]))

    #make on function DataFrame that has flag points
    # datas.append(getSimpleDataFrames())
    for d in getMoreDataFrames():
        datas.append(d)

    #make ONE non-function DataFrame

    #return list of DataFrames
    return datas



########################################################################
########################################################################
########################################################################
import unittest
class TestIsFunc(unittest.TestCase):

    dfs = getDataFrames()
        #first 3 are chemical reactions
        #4th is the trivial straight lines
        #5th is a data frame of quadratic polynomials 

    def testIsFunc(self):
        self.assertEqual(True, isF.isFunc(dfs[0])) #first order reaction
        self.assertEqual(True, isF.isFunc(dfs[1])) #first order reaction
        self.assertEqual(True, isF.isFunc(dfs[2])) #third order reaction
        self.assertEqual(False, isF.isFunc(dfs[3])) #clean, straight lines
        self.assertEqual(True, isF.isFunc(dfs[4])) #noisy polynomials


class TestFlagShelf(unittest.TestCase):
    """
    Test methods contained in FlagShelf.class
    """

    dfs = getMoreDataFrames()
    flaggy = isF.FlagShelf(dfs[0])

myTester = TestIsFunc()
myTester.testIsFunc()
