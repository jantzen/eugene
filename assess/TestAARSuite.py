#import numpy as np  ...

import eugene as eu

import AccuracyAndRobustnessSuite as AARS




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

    #grabbing values explicit in GrowthDemo.py -> Line 31 & 32
    #LGModel().__init__(self, r, init_x, K, alpha, beta, gamma, init_t)
    s_rs = [0.5, 1.0]
    s_ks = [65., 65., 65.]  #.... this part is fishy...
    #going to force a different K value for the third system...
    s_ks = [65., 65., 62.]

    classes1 = AARS.LGExperiment(n_std, s_rs, s_ks)
    #assert()....
    print type(classes1[0]), "==", type(eu.categorize.Classify)
    ###
