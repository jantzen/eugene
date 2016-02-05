import numpy as np
import matplotlib.pyplot as plt

import eugene as eu
#####
#Functions:
#  isFunc()
########################################################################

########################################################################
def isFunc(df):
    """
    df = a DataFrame object <eugene.interface.DataFrame>

    return True if 'df' is provides for symmetry transformations that are 
    functions.
       """
    
    #-1. make sure no 'outofrange' values in df.

   
    #0.model data
    #1.create modeled target data w/ respect to actual index data
    #2.if any modeled target value occurs more than once, record those points
    #   as "flag points".
    #3.find the target values of the first transformation at the index values
    #   of the "flag points"
    #4.iff the transformation values are equivalent, then the symmetry 
    #   transformations ARE FUNCTIONS.
    
    
    #0.
    pModels = []
    for tv in df._target_values:
        data = np.vstack((df._index_values, tv))
        data = data.transpose()
        pModels.append(eu.compare.FitPolyCV(data))
        
       
    #1.
    mData = [] #len(mData) = 3    .. should be
    for m in pModels:
        mata = []
        for iv in df._index_values:
            mata.append(np.polyval(m, iv))
            #clean mata w/ np.around()
            #try to only have 4 significant figures.
        mData.append(mata)
    
    #model Data Frame
    mDF = eu.interface.DataFrame(1, df._index_values, 2, mData)

    #2.
    # getFlagPoints(mData)
    #flag points = points which have same target value, but different index_v
    #if there are no flag points, then the DataFrame is a function.
    #if there is one+ flag points, then the DataFrame may or may not be a func!
    flags = []
    for i in range(len(mDF._target_values)):
        tv = mDF._target_values[i]
        if(len(np.unique(tv)) < len(tv)):
            print "found one flag! tv", i
            print len(np.unique(tv)), len(tv), '\n'
        
    out = False
    if(len(flags) == 0):
        out = True

    return out
