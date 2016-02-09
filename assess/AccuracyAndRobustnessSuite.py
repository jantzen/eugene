# import numpy as np
# import matplotlib.pyplot as plt

import eugene as eu


########
#Functions:
#
#
#
#######
#Classes:
#
#
#

#####################################################################
#####################################################################

def LGExperiment(noise_stdev=0):
    """
    ..(For now) it is assumed we use only three total systems:
      two K1 systems & one K2 system.
    """
    con = 0 #CONFUSED


    #index sensor & actuator
    isensor = eu.sensors.VABTimeSensor([])
    iact = eu.actuators.VABVirtualTimeActuator()
    #target sensor & actuator
    tsensor = eu.sensors.PopulationSensor([con, con], noise_stdev, False)
    tact = eu.actuators.PopulationActuator([con, con])
    
    sensors = dict([(1, isensor), (2, tsensor)])
    actuators = dict([(1, iact), (2, tact)])


#####################################################################
