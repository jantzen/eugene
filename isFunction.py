import pdb

import numpy as np
from scipy.interpolate import splrep, splev
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

    THOUGHTS---------------------------------
              1. bottom shelf has NO repeated values,
                 a. other shelves DO NOT have repeated values
                 b. other shelves DO     have repeated values

              2. bottom shelf HAS repeated values (2 instances of 1 value),
                 a. other shelves DO NOT have repeated values
                 b. other shelves DO     have repeated values, corresponding.
                 c. other shelves DO     have repeated values, uncorresponding.

              3. bottom shelf HAS repeated values (2 instances of 1 value
                                             & 3 instances of 1 other value),
                 a. other shelves DO NOT have repeated values
                 b. other shelves DO     have repeated values, corresponding.
                 c. other shelves DO     have repeated values, uncorresponding.

    """
    def __init__(self, df):
        self._df = df
        self._num_of_shelves = len(self._df._target_values)
        self._spline_df = None
        self._err = None

        #holds indexes of repeated values.
        self._shelves = {tv : [] for tv in range(self._num_of_shelves)}

        #find flags for each shelf
        self.fillBottomShelf()

    ##------------------------------------------------------------------
    """
    AM I SUPPOSED TO KEEP TRACK OF index values OR index locations!?
    right now, keeping track of index locations.
    """
    def fillBottomShelf(self):
        """
        Flag any concerning spots at the bottom of _target_values, 
        i.e. the domain of a possible
        symmetry function.
        """
        #Make a spline-function of the empirical df

        #best spline: scipy.interpolate.splrep

        # [[alternatively: scipy.interpolate.UnivariateSpline]]

        spline = splrep(self._df._index_values, 
                              self._df._target_values[0])


        # [[alternatively: a different cubic spline]]
        # [[spline = scipy.interpolate.UnivariateSpline(self._df._index_values, self._df._target_values[0],k=3)]]
                              
                                      #how do I fix the density of
                                      #values in bottomSpline?


        #determine self._mse from bottomSpline prediction wrt actual.
        #scipy.interpolate.splev


        # spline_values = splev(self._df._index_values, spline)
        # self._err = 0.01 #arbitrary..
        self._err = self.determineError(spline)
        

        #keep track of repeated values in _target_values[0], by means of
        #the index values XOR index locations!?!?
        #index locations != df._index_values
        for i, indVal in enumerate(self._df._index_values):
            repeated = [indVal]
            tarVal = splev(indVal, spline)

            for compIndVal in self._df._index_values[i+1:]:
                compTarVal = splev(compIndVal, spline)

                if( abs(tarVal - compTarVal) <= self._err):
                    repeated.append(compIndVal)
            #if we actually found any repeated values,
            # then store their locations.
            if (len(repeated) > 1):
                self._shelves[0].append(repeated)

    # determine the error for the spline function
    # [[if we use the other method we could use scipy.interpolate.UnivariateSpline.get_residual]]

    def determineError(self, spline):
        """
        Determines the error for the spline function for the bottom shelf.
        @param spline The spline previously determined for the bottom shelf.
          Only required as parameter because it's not a class attribute.
        """
        errors = []
        for indLoc, indVal in enumerate(self._df._index_values):
            diff = abs(self._df._target_values[0][indLoc]
                       - splev(indVal, spline))
            errors.append(diff)
        average = sum(errors) / len(errors)
        return average
        
       
    def fillEntireShelf(self):
        """
        Fills the rest of the shelves.

        return True when shelf fills without inconsistencies
        return False when shelf doesn't do so.
        """
        # if (len(self._shelves[0]) > 0):
        #done implicitly
        for shelfN, shelf in enumerate(self._shelves[1:]):
            #shelfN begins at 0. but should begin at 1.
            #therefore, always use "shelfN + 1" for using shelfN
            for flag in self._shelves[0]:
                fVals = []
                for indLoc in flag:
                    fVals.append(self._df._target_values[shelfN+1][indLoc])

def isFunc(df):
    """
    df = a DataFrame object <eugene.interface.DataFrame>

    return True if 'df' is provides for symmetry transformations that are 
    functions.
    """
    
    flaggy = FlagShelf(df)
    if (flaggy.bottomIsEmpty()):
        return True
    else:
        flaggy.fillShelf()
        if (flaggy.allShelvesLineUp()):
            return True
        else:
            return False
        
