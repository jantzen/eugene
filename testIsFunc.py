import numpy as np
import matplotlib.pyplot as plt
import math

import eugene as eu
import eugene.src.virtual_sys.chemical_sys as cs
from isFunction import isFunc

########################################################################
########################################################################
#create function data sets
def getDataFrames():
    """
       make 3 data samples that is a function.
       make 1 (3) data samples that is not a function.
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

    #make ONE non-function data set

    #return list of DataFrames
    return datas


def debugDF(df):
    """
       sometimes the values in df._target_values = 'outofrange'
       which is <type numpy.string_>
    """
    #print x,y if type(df._target_values[x][y]) == str

########################################################################
########################################################################
########################################################################

#Three cases to test for:
#   1. no flag points -> this is a function
#   2. 1 pair of flag points -> parallel points equal -> this is a function
#   3. 1 pair of flag points -> parallel points UNequal -> NOT a function

#right now, only case 1 is covered
def testIsFunc():
    dfs = getDataFrames()

    for df in dfs:
        assert True, isFunc(df)

        
