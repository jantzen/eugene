import numpy as np
import matplotlib.pyplot as plt

import eugene as eu
#####
#Classes:
#  FlagCabinet
#
#Functions:
#  getFlagShelf()
#  to4SigFig()
#  isFunc()
########################################################################

########################################################################
class FlagShelf( object ):
    """
    Flag Shelf contains 1 Shelf for each target value array in
         df._target_values.
    & contains 1 flags for each repeated value in a target value array.
    a flags = list of flag lists.
    flag list = list of indexes which have the same target value.
    """
    def __init__(self, df):
        self._df = df
        self._num_of_shelves = len(self._df._target_values)
        #other simple variables...
        #shelves
        self._shelves = {tv : [] for tv in range(self._num_of_shelves)}

        #find flags for each shelf
        for s in range(self._num_of_shelves):
            tv = df._target_values[s]
            for ind, item in enumerate(tv):
                flag = [ind]
                for c, checks in enumerate(tv[ind:]):
                    if ((not c == 0) and item == checks):
                        flag.append(c+ind)
                if (len(flag) > 1):
                    self._shelves[s] == flag
                        
        
    

def getFlagShelf(df):
    """
    given a data frame,
    returns a dictionary 'flagShelf'

    number of keys in flagShelf = number of target value arrays
    each value defaults as empty list
    if there are duplicate values in a given _target_value array,
    the INDEX of those duplicates are saved.
    """
    #create an entry for every _target_values array
    flagShelf = {tv : [] for tv in range(len(df._target_values))}
    
    #dict's key  
    k = 0
    for tv in df._target_values:
        for i in range(len(tv)):
            #c = compare index
            for c in range(len(tv[i:])):
                if (tv[i] == tv[c]):
                    flagShelf(k).append[i,c]

        k += 1
                    
                    

    return flagShelf


def to4SigFigs(x):
    """
    x = float
    appropriately round x to have 4 sigfigs.

    returns x as 4 sigfigs
    """
    
    s = str(x)
    
    #i'll assume any float with a 'NNNNNe-XX'
    #will have 'e' at location [-4]
    if ('e' in s):
        number = s[:-4]
        exponent = s[-4:]
        FourSigFig = np.around(float(number), 4)
        cleanX = float(str(FourSigFig) + exponent)
    else:
        FourSigFig = np.around(x, 4)
        cleanX = FourSigFig

    return cleanX

def isFunc(df):
    """
    df = a DataFrame object <eugene.interface.DataFrame>

    return True if 'df' is provides for symmetry transformations that are 
    functions.
       """
    
    #-1.preprocess df: make sure no 'outofrange' values

   
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
        #comment: get FitPolyCV to spit out standard deviation
        
       
    #1. -- change: until flag point is found, only model & mDF the base values
                   #for computational efficiency.
    mData = []
    for m in pModels:
        mata = []
        for iv in df._index_values:
            mata.append(to4SigFigs(np.polyval(m, iv)))
            #------use Standard Deviation instead
            #clean mata w/ np.around()
            #try to only have 4 significant figures.
        mData.append(mata)

    
    #model Data Frame
    mDF = eu.interface.DataFrame(1, df._index_values, 2, mData)

    #2.
    # getFlagPoints(mData)
    #flag points = points which have same value w/in a _tar_val array, 
    #              but different index_v
    #if there are no flag points, then the DataFrame is a function.
    #if there is one+ flag points, then the DataFrame may or may not be a func!
    fShelf = FlagShelf(mDF)

        
    hasFlags = False
    if(len(flags) > 0):
        hasFlags = True


    return hasFlags
