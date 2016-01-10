# from func import getDataSets
import numpy as np
import eugene as eu

import matplotlib.pyplot as plt
########################################################################

#should this only consider 'is a function?' for a system with respect to only
# one sample in one dataframe? (prolly not?)
def isFunc(ds):
    """
    ds = sample of data with only one set of y-x points

    return True if 'ds' is a data set describing a function
       """


    #######################################
    #create '2' models- as if 'ds' is a function, and another as if it isn't
    asFunc = eu.compare.FitPolyCV(ds)
    asNotF1 = None
    asNotF2 = None

    #make 'not function part 1' and 2; an ndarray, not a list
    nF1data = np.array([[]])
    nF2data = np.array([[]])
    #find mean target value. - to split ds into nF1 and nF2
    mean = np.mean(ds[:,1])


    #for each target value > mean, add to nF1data
    # if < mean, then add to nF2data
    for s  in ds:
        indx, tar = s[0], s[1]
        if (tar > mean):
            nF1data = np.append(nF1data, [indx, tar])
        else:
            nF2data = np.append(nF2data, [indx, tar])
    nF1data = np.reshape(nF1data, (-1, 2))
    nF2data = np.reshape(nF2data, (-1, 2))
    
    asNotF1 = eu.compare.FitPolyCV(nF1data)
    asNotF2 = eu.compare.FitPolyCV(nF2data)


    ###################################################
    #which model is better?  !!
    #  1st, find MSE for both
    erFunc = []
    erNfnc = []

    for x in ds[:,0]:
        erFunc.append(abs(x - np.polyval(asFunc, x)))

        if x in nF1data:
            erNfnc.append(abs(x - np.polyval(asNotF1, x)))
        else:
            erNfnc.append(abs(x - np.polyval(asNotF2, x)))

    isFn = None
    
    ############
    print "mean"
    print "\t func", np.mean(erFunc)
    print "\t notF", np.mean(erNfnc)
    print "sum"
    print "\t func", sum(erFunc)
    print "\t notF", sum(erNfnc)
    print "result:"
    # if simpleEr_func > simpleEr_not:
    #     isFn = True
    # else:
    #     isFn = False
    # return isFn

    ###################################################    
    #print the models to the dataset

    # for p in ds:
    #     plt.scatter(p[0], p[1], c='b')
    # plt.show()

    # for x in ds[:,0]:
    #     m = [x, np.polyval(asFunc, x)]
    #     plt.scatter(m[0], m[1], c='g')
    #     if x in nF1data:
    #         m = [x, np.polyval(asNotF1, x)]
    #         plt.scatter(m[0], m[1], c='r')
    #     else:
    #         m = [x, np.polyval(asNotF2, x)]
    #         plt.scatter(m[0], m[1], c='r')
    # plt.show()
    ##############################


    #######################
    #return *bool*
    s = None #bool for sum() evaluation
    m = None # bool for mean() evaluation
    if sum(erFunc) < sum(erNfnc):
        s = True
    else:
        s = False
    if np.mean(erFunc) < np.mean(erNfnc):
        m = True
    else:
        m = False

    print "sum", s, "\nmean", m
    print "------------------------"

    #if isFunc(DS) == True, then DS "is a function"
    return (s and m) #or evaluate bool variables & return "isFn" from line60
