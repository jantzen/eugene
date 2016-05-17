# import matplotlib.pyplot as plt
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
    stdev = 18.
    skews = np.arange(0., 30., 3.)

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
            else: 
                raise ValueError("output from test not binary")

        CM[k] = [[TP, FP], [FN, TN]]
                
    return data, CM


def GrowthRateExperiment(samples, free_cores=1):
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
    rvals = np.arange(0.1, 10.1, 1.)

    
    confusion_matrices = dict()

    data = dict()
    CM = dict()
    for r in rvals: 
        # set parameters of Logistic Growth models to test
        systems = [[1, 1, 100, 1, 1, 1, 0], [r, 1, 100, 1, 1, 1, 0]]
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
        data[r] = []
        for job in jobs:
            data[r].append(job())

        # compile confusion matrix for this noise level
        CM[r] = []
        for entry in data[r]:
            if entry == [1, 1]:
                TP += 1
            elif entry == [1, 0]:
                FP += 1
            elif entry == [0, 0]:
                TN += 1
            elif entry == [0, 1]:
                FN += 1
            else: 
                raise ValueError("output from test not binary")

        CM[r] = [[TP, FP], [FN, TN]]
                
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


def LGExperiment(noise_stdev, sp1, sp2, skew=0):
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
                                          False, skew)
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

    # determine the correct answer (assuming distinct r-values are equivalent)
    if sp1[1:] == sp2[1:]:
        true_answer = 0
    else :
        true_answer = 1

    return [decision, true_answer]



def CircuitNoiseExperiment(samples, free_cores=1):
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
    stdevs = np.arange(2.05, 4.0, 0.2)

    # set names of chaotic models to test
    systems = [7, 8]
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
                jobs.append(job_server.submit(CircExperiment,(noiselevel,sys[0],sys[1]),(),("eugene",)))
        
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
 


def CircExperiment(noise_stdev, sp1, sp2, skew=0):
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
    resolution=[600,1]
    ###

    ###
    #index sensor & actuator
    isensor = eugene.sensors.VABTimeSensor([])
    iact = eugene.actuators.VABVirtualTimeActuator()

    #target sensors & actuators
    tsensor0 = eugene.sensors.CCVoltageSensor([-10**23,10.**23], 0, noise_stdev, False)
    tsensor1 = eugene.sensors.CCVoltageSensor([-10**23,10.**23], 1, noise_stdev, False)
    tsensor2 = eugene.sensors.CCVoltageSensor([-10**23,10.**23], 2, noise_stdev, False)
    tact0 = eugene.actuators.CCVoltageActuator([0.,10.**23], 0)
    tact1 = eugene.actuators.CCVoltageActuator([0.,10.**23], 1)
    tact2 = eugene.actuators.CCVoltageActuator([0.,10.**23], 2)
 
    

    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(3, isensor), (0, tsensor0), (1, tsensor1), (2, tsensor2)])
    actuators = dict([(3,iact),(0,tact0), (1,tact1), (2,tact2)])

    # build systems from data
    systems = []

    systems.append(eugene.chaotic_circuits.ChaoticCircuit(sp1))
    systems.append(eugene.chaotic_circuits.ChaoticCircuit(sp2))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(eugene.interface.VABSystemInterface(sensors, actuators, sys))

    # build ROIs
    ROI = []
    ROI.append(dict([(3,[0., 10.]), (0, [1.763, -0.6444085]), (1,
        [0.655697, 2.14696]), (2, [-0.140532, 0.258348])]))
    ROI.append(dict([(3,[0., 10.]), (0, [1.763, -0.6444085]), (1,
        [0.655697, 2.14696]), (2, [-0.140532, 0.258348])]))
 
 
    ### collect data
    data = []
    for count, iface in enumerate(interfaces):
        data.append(eugene.interface.TimeSampleData(3, [0,1,2], iface,
            ROI[count], resolution, True))
    ###

    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(eugene.compare.BuildSymModel(data_frame, 3, [0,1,2], sys_id, epsilon))

    decision = eugene.compare.CompareModels(models[0], models[1])

    # determine the correct answer
    if sp1 == sp2:
        true_answer = 0
    else :
        true_answer = 1

    return [decision, true_answer]


#####################################################################

def LVNoiseExperiment(samples, free_cores=1):
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
    stdevs = np.arange(0.5, 30.5, 3.)

    # set parameters for models to test
    params = [[2., 4., 100., 100., 1., -1., 5., 5.],[3., 3., 100., 100., 1., -1., 5., 5.]]
    system_combos = []
    for i in range(2):
        for j in range(2):
            system_combos.append([params[i],params[j]])

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
            for params in system_combos:
                jobs.append(job_server.submit(LVExperiment,(noiselevel,params[0],params[1]),(),("eugene",)))
        
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
 


def LVExperiment(noise_stdev, sp1, sp2, skew=0):
    """
    @short Description--------------------------------
      Runs simulated Lotka-Volterra experiments for Biological Populations.
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
    resolution=[600,1]
    ###

    ###
    #index sensor & actuator
    isensor = eugene.sensors.VABTimeSensor([])
    iact = eugene.actuators.VABVirtualTimeActuator()

    #target sensors & actuators
    tsensor1 = eugene.sensors.LotkaVolterra2DSensor(1, [-10**23,10.**23], noise_stdev, False)
    tsensor2 = eugene.sensors.LotkaVolterra2DSensor(2, [-10**23,10.**23], noise_stdev, False)
    tact1 = eugene.actuators.LotkaVolterra2DActuator(1, [-10**23.,10.**23])
    tact2 = eugene.actuators.LotkaVolterra2DActuator(2, [-10**23.,10.**23])


    #build a dictionary of sensors and a dictionary of actuators
    sensors = dict([(0, isensor), (1, tsensor1), (2, tsensor2)])
    actuators = dict([(0,iact), (1,tact1), (2,tact2)])

    # build systems from data
    systems = []

    systems.append(eugene.LotkaVolterra2D.LotkaVolterra2D(sp1[0],sp1[1],sp1[2],sp1[3],sp1[4],sp1[5],
        sp1[6],sp1[7]))
    systems.append(eugene.LotkaVolterra2D.LotkaVolterra2D(sp2[0],sp2[1],sp2[2],sp2[3],sp2[4],sp2[5],
        sp2[6],sp2[7]))

    # build corresponding interfaces
    interfaces = []
    for sys in systems:
        interfaces.append(eugene.interface.VABSystemInterface(sensors, actuators, sys))

    # build ROIs
    ROI = dict([(0, [0., 1.]), (1, [1., 100.]), (2, [1., 100.])])

 
    ### collect data
    data = []
    for count, iface in enumerate(interfaces):
        print "Sampling data for system {}. ROI for time: {}. ROI for x1: {}.  ROI for x2: {}.\n".format(count, ROI[0], ROI[1], ROI[2])
        data.append(eugene.interface.TimeSampleData(0, [1,2], iface,
            ROI, resolution, True))
    ###

    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(eugene.compare.BuildSymModel(data_frame, 0, [1,2], sys_id, epsilon))

    decision = eugene.compare.CompareModels(models[0], models[1])

    # determine the correct answer
    if sp1 == sp2:
        true_answer = 0
    else :
        true_answer = 1

    return [decision, true_answer]

