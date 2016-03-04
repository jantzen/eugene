import eugene as eu
import pdb
import sys
import random
import numpy as np

import eugene as eu
from eugene.src.virtual_sys.growth_sys import LogisticGrowthModel
import scipy as sp


def SimilarityExperimentFrame():
    # tests performance as the models become increasingly dissimilar
    # alterations to gamma; varied between 1-2
    # r = 1, gamma =1 vs r=1, gamma=2
    
    gammas = [1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
    sys1 = [1, 1, 50, 1, 1, 1, 0]
    sys2 = [1, 1, 50, 1, 1, 1, 0]
    twoSys = [sys1, sys2]

    SIM = dict()
    for gamma in gammas:
        answers = []
        #bracket each of eugene's answers
        predictions = []
        #10 trials per similarity level
        for x in range(10):
            first = random.randint(0,1)
            second  = random.randint(0,1)
            if (first == second):
                answers.append(0)
            else:
                answers.append(1)
            predictions.append(SimilarityExperiment(gamma, 
                                        twoSys[first],
                                        twoSys[second]))
        

        correct = 0
        ans = 0
        for i in range(len(answers)):
            ans += 1
            if (answers[i] == predictions[i]):
                correct += 1
        score = correct / ans
        SIM[gamma] = [answers, predictions, score]

    return SIM


def SimilarityExperiment(gamma, sp1, sp2):
    """
    @short Description--------------------------------
      Runs simulated LogisticGrowth experiments for Biological Populations.
      <Returns meaningful data>
"""

    ### local variables
    con = 5.0     
    epsilon=10**(-4)
    resolution=[300,3]
    ###

    ###
    #index sensor & actuator
    isensor = eu.sensors.VABTimeSensor([])
    iact = eu.actuators.VABVirtualTimeActuator()
    #target sensor & actuator

    tsensor = eu.sensors.PopulationSensor([-10**(23), 10**23], 1, 
                                          False)
    tact = eu.actuators.PopulationActuator([0, 10**23])
    
    sensors = dict([(1, isensor), (2, tsensor)])
    actuators = dict([(1, iact), (2, tact)])
    ###

    ###
    systems = []
    #LGModel().__init__(self, r, init_x, K, alpha, beta, gamma, init_t)
    systems.append(LogisticGrowthModel(sp1[0], sp1[1], sp1[2], sp1[3],
                                       sp1[4], sp1[5], sp1[6]))


    systems.append(LogisticGrowthModel(sp2[0], sp2[1], sp2[2], sp2[3],
                                       sp2[4], gamma, sp2[6]))


    interfaces = []
    for sys in systems:
        interfaces.append(eu.interface.VABSystemInterface(sensors, actuators, sys))
    ROI = dict([(1,[0., 20.]), (2, [con, 65.])])
    ###    


    ### collect data!
    data = []
    for count, iface in enumerate(interfaces):
        data.append(eu.interface.TimeSampleData(1, 2, iface, ROI, resolution))
    ###

    ### Line143-144 is where it breaks
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(BuildModel(data_frame, sys_id, epsilon))

    same0diff1 = eu.compare.CompareModels(models[0], models[1])

    return same0diff1
