# import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'C\vabacon')
import eugene as eu
import numpy as np

"""Accuracy and Robustness Suite

Sets up four experiments demonstrating accuracy and robustness: one showing 
performance under increasing levels of Gaussian noise; one showing performance 
under noise with increasing deviations from normality; one showing performance 
as it varies with increasing similarity among systems; one showing performance
as it changes with variation in r-values versus k values.""" 

# establish values for r's and k's
systems_rs = np.ndarray(.5,1,2)
systems_ks = np.ndarray(50, 100)

########
#Functions:

def SimpleNoiseExperiment():
   # basic noise experiment
    standard_devs = [0, 1., 2., 4., 6., 8., 10.] 
    for noiselevel in standard_devs: 
        return(LGExperiment(noiselevel, systems_rs, systems_ks))
           
        
def FullNoiseExperiment(r_range, k_pairs):   
    # vary which pairs we're using so we can have different kinds of 
    # systems we're comparing
    # inside this, run experiment
    pass
#Classes:
#
#
#
#####################################################################
#####################################################################

def LGExperiment(noise_stdev, systems_rs, systems_ks):
    # re: below - It doesn't have to be quail. I just liked looking at papers
    # about quail. - Jack
    """
    @short Description--------------------------------
    Runs simulated LogisticGrowth experiments for biological 
    populations.
    <Returns meaningful data>

    @params:------------------------------------------
    noise_stdev reflects the (square root of the) variance of our Gaussian
    distribution. In a noise experiment, we steadily increase this value. 

    systems_rs Is an np.ndarray (assumed: size = 3). The element at each 
    index corresponds to each of the three systems. An "r" is not a variable
    that differentiates (these kinds of) natural kinds.

    systems_ks Is an np.ndarray (assumed: size = 2). The first element is
    the K value for one kind of LogisticGrowthModel. The second element 
    is the K value for the next kind of model. (This is the one on which
    we expect different kinds; AFAIK different carrying capacities reflect very 
    different systems. - Jack)

    @return-------------------------------------------
    ....meaningful data.... list of classes determined out of the
    three simulated systems.

    """


    ### local variables
    con = 0.0 #CONFUSED    (where does Janzten get these numbers from??)
    epsilon=10**(-4)
    resolution=[300,3]
    alpha=1
    ###

    systems_rs = np.ndarray(.5,1,2)
    systems_ks = np.ndarray(50, 100)

    ###
    #index sensor & actuator
    isensor = eu.sensors.VABTimeSensor([])
    iact = eu.actuators.VABVirtualTimeActuator()
    #target sensor & actuator
    tsensor = eu.sensors.PopulationSensor([con, con], noiselevel, False)
    #tsensor = eu.sensors.PopulationSensor([-10.**23, 10.**23], noise_stdev, 
    #                                      False)   ^XOR ?
    tact = eu.actuators.PopulationActuator([con, con])
    
    sensors = dict([(1, isensor), (2, tsensor)])
    actuators = dict([(1, iact), (2, tact)])
    ###

    ###
    systems = []
    #LGModel().__init__(self, r, init_x, K, alpha, beta, gamma, init_t)
    systems.append(LogisticGrowthModel(systems_rs[0], 1, 1, 1, 1, 1, 1))
    systems.append(LogisticGrowthModel(systems_rs[0], 1, 1, 1, 1, 1, 1))
    systems.append(LogisticGrowthModel(systems_rs[1], 1, 1, 1, 1, 1, 1))
    
    interfaces = []
    for sys in systems:
        interfaces.append(eu.interface.VABSystemInterface(sensors, actuators, sys))
    #blaahhhh - in case you were wondering how we find ROI, we're not 
    #           supposed to: this is where Collin's AutoRangeDet thing 
    #           comes in.
    ROI = dict([(1,[con, 20.]), (2, [con, 65.])])
    ###    


    ### collect data!
    data = []
    for count, iface in enumerate(interfaces):
        data.append(eu.interface.TimeSampleData(1, 2, iface, ROI, resolution))
    ###

    ### Eugene, use your magic powers of witchcraft & wizardry! :P hah...
    models = []
    for sys_id, data_frame in enumerate(data):
        models.append(BuildModel(data_frame, sys_id, epsilon))

    classes = eu.categorize.Classify(range(len(systems)), models)
    ### Hell yeah!

    return classes

#####################################################################
