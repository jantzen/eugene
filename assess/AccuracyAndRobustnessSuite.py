# import matplotlib.pyplot as plt
import pdb
#import sys
import random
import numpy as np
import scipy 
import pp

#sys.path.insert(0, 'C\vabacon')
import eugene 

"""Accuracy and Robustness Suite

Sets up four experiments demonstrating accuracy and robustness: one showing 
performance under increasing levels of Gaussian noise; one showing performance 
under noise with increasing deviations from normality; one showing performance 
as it varies with increasing similarity among systems; one showing performance
as it changes with variation in r-values versus k values.""" 

########
#Functions:
#  SkewNorm
#  SimpleNoiseExperiment
#  DeviantNoiseExperiment
#  FullNoiseExperiment
#  LGExperiment


def GrowthNoiseExperiment(samples, free_cores=1):
    """ sample = number of samples to take of each comparison type (four types
    possible)
        free_cores = minimum number of cores to leave unloaded
    """
    # set up job server for pp
    job_server = pp.Server()

    # determine how many cores to use
    cpus = max(1, job_server.get_ncpus() - free_cores)

    job_server.set_ncpus(cpus)


    # set noise levels to test
    stdevs = np.arange(3.,33.,3.)

    # set parameters of Logistic Growth models to test
    systems = [[1, 1, 100, 1, 1, 1, 0], [1, 1, 90, 1, 1, 1, 0]]
    system_combos = []
    for i in range(2):
        for j in range(2):
            system_combos.append([systems[i],systems[j]])

    confusion_matrices = dict()

    data = dict()
    CM = dict()
    for noiselevel in stdevs: 
        # set up variables to store elements of confusion matrix
        # positive case: systems are different
        # negative case: systems are the same
        TP = 0 # [1,1]
        FP = 0 # [1,0]
        TN = 0 # [0,0]
        FN = 0 # [0,1]

        jobs = []
 
        for i in range(samples): #trials per noise level
            for sys in system_combos:
                jobs.append(job_server.submit(LGExperiment,(noiselevel,sys[0],sys[1]),(),("eugene",)))
        
        # gather the data
        data[noiselevel] = []
        for job in jobs:
            data[noiselevel].append(job())

        # compile confusion matrix for this noise level
        CM[noiselevel] = []
        for entry in data[noiselevel]:
            if entry == [1, 1]:
                TP += 1
            elif entry == [1, 0]:
                FP += 1
            elif entry == [0, 0]:
                TN += 1
            elif entry == [0, 1]:
                FN += 1

        CM[noiselevel] = [[TP, FP], [FN, TN]]
                
    return data, CM
    

def GrowthNormDeviationExperiment(samples, free_cores=1):
    """ sample = number of samples to take of each comparison type (four types
    possible)
        free_cores = minimum number of cores to leave unloaded
    """
    # set up job server for pp
    job_server = pp.Server()

    # determine how many cores to use
    cpus = max(1, job_server.get_ncpus() - free_cores)

    job_server.set_ncpus(cpus)


    # set noise levels to test
    stdev = 5.
    skews = np.arange(10.)

    # set parameters of Logistic Growth models to test
    systems = [[1, 1, 100, 1, 1, 1, 0], [1, 1, 90, 1, 1, 1, 0]]
    system_combos = []
    for i in range(2):
        for j in range(2):
            system_combos.append([systems[i],systems[j]])

    confusion_matrices = dict()

    data = dict()
    CM = dict()
    for a in skews: 
        # set up variables to store elements of confusion matrix
        # positive case: systems are different
        # negative case: systems are the same
        TP = 0 # [1,1]
        FP = 0 # [1,0]
        TN = 0 # [0,0]
        FN = 0 # [0,1]

        jobs = []
 
        for i in range(samples): #trials per noise level
            for sys in system_combos:
                jobs.append(job_server.submit(LGExperiment,(stdev,sys[0],sys[1],a),(),("eugene",)))
        
        # gather the data
        data[a] = []
        for job in jobs:
            data[a].append(job())

        # compile confusion matrix for this noise level
        CM[a] = []
        for entry in data[a]:
            if entry == [1, 1]:
                TP += 1
            elif entry == [1, 0]:
                FP += 1
            elif entry == [0, 0]:
                TN += 1
            elif entry == [0, 1]:
                FN += 1

        CM[a] = [[TP, FP], [FN, TN]]
                
    return data, CM


def GrowthModelExperiment(samples, free_cores=1):
    """ sample = number of samples to take of each comparison type (four types
    possible)
        free_cores = minimum number of cores to leave unloaded
    """
    # set up job server for pp
    job_server = pp.Server()

    # determine how many cores to use
    cpus = max(1, job_server.get_ncpus() - free_cores)

    job_server.set_ncpus(cpus)

    # set k levels to test
    kvals = np.arange(95.,100.,0.5)

    
    confusion_matrices = dict()

    data = dict()
    CM = dict()
    for k in kvals: 
        # set parameters of Logistic Growth models to test
        systems = [[1, 1, 100, 1, 1, 1, 0], [1, 1, k, 1, 1, 1, 0]]
        system_combos = []
        for i in range(2):
            for j in range(2):
                system_combos.append([systems[i],systems[j]])
        
        # set up variables to store elements of confusion matrix
        # positive case: systems are different
        # negative case: systems are the same
        TP = 0 # [1,1]
        FP = 0 # [1,0]
        TN = 0 # [0,0]
        FN = 0 # [0,1]

        jobs = []
 
        for i in range(samples): #trials per noise level
            for sys in system_combos:
                jobs.append(job_server.submit(LGExperiment,(3,sys[0],sys[1]),(),("eugene",)))
        
        # gather the data
        data[k] = []
        for job in jobs:
            data[k].append(job())

        # compile confusion matrix for this noise level
        CM[k] = []
        for entry in data[k]:
            if entry == [1, 1]:
                TP += 1
            elif entry == [1, 0]:
                FP += 1
            elif entry == [0, 0]:
                TN += 1
            elif entry == [0, 1]:
                FN += 1

        CM[k] = [[TP, FP], [FN, TN]]
                
    return data, CM
 
#def DeviantNoiseExperiment():
#    #tests performance as noise distribution departs from normality
#    #the following are alpha/skew levels for a skew normal distribution,
#    skew_levels = [0, 1., 2., 3., 4., 5.]
#    sys1 = [1, 1, 60, 1, 1, 1, 0]
#    sys2 = [1, 1, 65, 1, 1, 1, 0]
#    twoSys = [sys1, sys2]
#
#    DNE = dict()
#    for skew_level in skew_levels:
#        answers = []
#        #bracket each of eugene's answers
#        predictions = []
#        #10 trials per noise level
#        for x in range(10):
#            first = random.randint(1,2)
#            second  = random.randint(1,2)
#            predictions.append(LGExperiment(st_dev=1,skew_level, 
#                                        twoSys[first],
#                                        twoSys[second]))
# # Show results using same strategy as SimpleNoiseExperiment                                     
#        correct = 0
#        numOfAns = 0
#
#        for i in range(len(answers)):
#            numOfAns += 1
#
#            if (answers[i] == predictions[i]):
#                correct += 1
#
#        if (numOfAns > 0):
#            score =  correct / float(numOfAns)
#
#        else:
#            score = None            
#
#        DNE[skew_level] = [answers, predictions, score]
#
#    return DNE


def LGExperiment(noise_stdev, sp1, sp2,skew=0):
    """
    @short Description--------------------------------
      Runs simulated LogisticGrowth experiments for Biological Populations.
      <Returns meaningful data>

    @params:------------------------------------------
      noise_stdev reflects the (square root of the) variance of our Gaussian
      distribution. In a noise experiment, we steadily increase this value. 

      sp1 = system parameters 1.
      sp2 = system parameters 2.

    @return-------------------------------------------
      ....meaningful data.... list of classes determined out of the
      three simulated systems.

    """

    ### local variables
    epsilon=10**(-4)
    resolution=[300,3]
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
    
    ROI = dict([(1,[0., 20.]), (2, [5., 55.])])
    ###    


    ### collect data!
    data = []
    for iface in interfaces:
        data.append(eugene.interface.TimeSampleData(1, [2], iface, ROI, resolution))
    ###

    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(eugene.compare.BuildSymModel(data_frame, 1, [2], sys_id, epsilon))

    decision = eugene.compare.CompareModels(models[0], models[1])

    # determine the correct answer
    if sp1 == sp2:
        true_answer = 0
    else :
        true_answer = 1

    return [decision, true_answer]

#####################################################################
