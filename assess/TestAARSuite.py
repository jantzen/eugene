#import numpy as np  ...

import eugene as eu

import AccuracyAndRobustnessSuite as AARS


AARS.SimpleNoiseExperiment()

def testLGExperiment():
    """
    test the LGExperiment method inside AccuracyAndRobustnessSuite.

    (transcendent?) variables:___________________________
    |test's Arguments:     |LGExperiment ( params )     |
    |----------------------|----------------------------|
    |              n_std   |->   noise_stdev            |
    |              s_rs    |->   systems_rs             |
    |              s_ks    |->   systems_ks             |
    |---------------------------------------------------|
   
    """

    ###
    n_std = 0

    #LGModel().__init__(self, r, init_x, K, alpha, beta, gamma, init_t)
    sys1 = [1, 1, 1, 1, 1, 1, 0]
    sys2 = [1, 1, 2, 1, 1, 1, 0]

    classes1 = AARS.LGExperiment(n_std, sys1, sys1)

    #assert()....
    # print type(classes1[0]), "==", type(eu.categorize.Classify)
    ###


