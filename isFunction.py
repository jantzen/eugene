import pdb

import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

import eugene as eu
#####
#Classes:
#  FlagShelf
#  DangerousFlagExcpetion
#
#Functions:
#  
#  isFunc()
########################################################################

########################################################################
#Tasks:
#   Test.
#   get splrep's residuals.
#   new targetValuesEqual(t1, t2) wrt to residuals.
#   Test.
########################################################################
class DangerousFlagException(Exception):
    """
    An exception which is thrown when a flag is discovered
    to be dangerous, i.e. implies that the symmetry structure cannot be
    defined as a function.
    """
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)




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
        self._err = None

        #holds indexes of repeated values.
        self._shelves = {tv : [] for tv in range(self._num_of_shelves)}
        self._spline_frame = {tv : None for tv in range(self._num_of_shelves)}
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
        self._spline_frame[0] = spline


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

                if(targetValuesEqual(tarVal, compTarVal)):
                    repeated.append(compIndVal)
            #if we actually found any repeated values,
            # then store their locations.
            if (len(repeated) > 1):
                self._shelves[0].append(repeated)

    def fillShelf(self, shelfN):
        """
        Fill a desired shelf with flags.
        
        @precondition: The bottom shelf has at least one flag.

        @param shelfN The desired shelf to fill.

        @throws DangerousFlagException when a flag is found which
         defines a point that 'makes the dataframe not-a-function.'
        """
        #double check that bottom shelf has at least one flag.
        assert len(self._shelves[0]) > 0
        
        spline = splrep(self._df._index_values, 
                        self._df._target_values[shelfN])
        self._spline_frame[shelfN] = spline


        for flag in self._shelves[0]:
            #flagged Target Value - for reference.
            reference_fTV = splev(flag[0], spline)
            newFlag = [reference_fTV]
            for flaggedIndVal in flag[1:]:
                tarVal = splev(flaggedIndVal, spline)
                if (targetValuesEqual(reference_fTV, tarVal)):
                    newFlag.append(tarVal)
                else:
                    raise DangeroutFlagException("target values don't align!")

            #this may be an unnecessary check
            if len(newFlag) > 0:
                self._shelves[shelfN].append(newFlag)



    def targetValuesEqual(tar_val1, tar_val2):
        """
        Status:
            Tenetatively implemented.
            Used in fillBottomShelf.

        Given two taret values, determine whether they are equal.
        """
        return abs(tar_val1 - tar_val2) <= self._err

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
        for n in range(1, flaggy._num_of_shelves):
            try:
                flaggy. fillShelf(n)
            except DangerousFlagException:
                return False
            
        return True
