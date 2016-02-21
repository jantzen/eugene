# import matplotlib.pyplot as plt
import pdb

import sys
import random
import numpy as np

# sys.path.insert(0, 'C\vabacon')
import eugene as eu
from eugene.src.virtual_sys.growth_sys import LogisticGrowthModel
import scipy as sp
from sp.special import pi,sqrt,exp

"""Accuracy and Robustness Suite

Sets up four experiments demonstrating accuracy and robustness: one showing 
performance under increasing levels of Gaussian noise; one showing performance 
under noise with increasing deviations from normality; one showing performance 
as it varies with increasing similarity among systems; one showing performance
as it changes with variation in r-values versus k values.""" 

def BuildModel(data_frame, sys_id, epsilon=0):
    model = eu.compare.BuildSymModel(data_frame, 1, 2, sys_id, epsilon)

    return model
#!!!!!!!! Change BuildSymModel params^^

########
#Functions:



def SimpleNoiseExperiment():
   # basic noise experiment
    standard_devs = [0.1, 1., 2., 4., 6., 8., 10.]
    sys1 = [1, 1, 1, 1, 1, 1, 0]
    sys2 = [1, 1, 2, 1, 1, 1, 0]
    twoSys = [sys1, sys2]

    noise_bracket = []
    for noiselevel in standard_devs: 
        #keep track of known answers
        answers = []
        #bracket each of eugene's answers
        bracket = []
        #10 trials per noise level
        #EUGENE (really numpy..) breaks at standard_devs >= 0.32 b/c
        #b/c target values = 'outofrange'
        for x in range(10):
            first = random.randint(0,1)
            second  = random.randint(0,1)
            if (first == second):
                answers.append(0)
            else:
                answers.append(1)
            bracket.append(LGExperiment(noiselevel, 
                                        twoSys[first],
                                        twoSys[second]))
            
        noise_bracket.append(bracket)
        #print results for developing
        correct = 0
        for i in range(len(answers)):
            if (answers[i] == bracket[i]):
                correct += 1
        print '\n' + 'noise:', noiselevel
        print 'answer:', answers
        print 'eugene:', bracket
        print 'correct/all: ', correct, '/', len(answers), '\n'

    #temporary return value
    return noise_bracket
           
# We need something like the thing below for the DeviantNoiseExperiment, under 
# construction obviously
class skewed_normal(sp.stats.rv_continuous):
    # Towards the sampler doodad, for grabbing a value from a SND    
    # Builds a skewed normal distribution
    def __init__(self, skew):
        self.skew = skew
        
    def pdf(x):
        # standard normal density function
        numerator = exp(-x**2/2)
        denominator = 1/sqrt(2*pi) 
        return numerator / denominator 
    
    def cdf(x):
        # distribution function for the standard normal
        return (1 + sp.special.erf(x/sqrt(2))) / 2
        
    def skewed(self, x, skew):
        # set up the pdf for the skewed normal distribution.
        # f(x) = 2 * pdf(x) * cdf(skew * x)   [skew is a shape parameter]
        
        return 2 * skewed_normal.pdf(x) * skewed_normal.cdf(x * skew)

        
def DeviantNoiseExperiment():
    #tests performance as noise distribution departs from normality
    #the following are alpha/skew levels for a skew normal distribution,
    # sensible ones will be added later
    skew_levels = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    sys1 = [1, 1, 1, 1, 1, 1, 0]
    sys2 = [1, 1, 2, 1, 1, 1, 0]
    twoSys = [sys1, sys2]

    snoise_bracket = []
    for skew_level in skew_levels:
        skewedgaussian = skewed_normal(self,skew, [args])    
        bracket = []
        first = random.randint(1,2)
        second  = random.randint(1,2)
        
        #10 trials per noise level
        for x in range(10):
            bracket.append(LGExperiment(skew_level, 
                                        twoSys[first],
                                        twoSys[second]))
        snoise_bracket.append(bracket)
        
    #temporary return value
    return snoise_bracket




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

def LGExperiment(noise_stdev, sp1, sp2):
    # re: below - It doesn't have to be quail. I just liked looking at papers
    # about quail. - Jack
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
    con = 5.0 #CONFUSED    (where does Janzten get these numbers from??)
    epsilon=10**(-4)
    resolution=[300,3]
    ###

    ###
    #index sensor & actuator
    isensor = eu.sensors.VABTimeSensor([])
    iact = eu.actuators.VABVirtualTimeActuator()
    #target sensor & actuator

    tsensor = eu.sensors.PopulationSensor([10**(-23), 10**23], noiselevel, 
                                          False)
    #tsensor = eu.sensors.PopulationSensor([-10.**23, 10.**23], noise_stdev, 
    #                                      False)   ^XOR ?
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
                                       sp2[4], sp2[5], sp2[6]))


    
    interfaces = []
    for sys in systems:
        interfaces.append(eu.interface.VABSystemInterface(sensors, actuators, sys))
    #blaahhhh - in case you were wondering how we find ROI, we're not 
    #           supposed to: this is where Collin's AutoRangeDet thing 
    #           comes in.
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

#####################################################################

