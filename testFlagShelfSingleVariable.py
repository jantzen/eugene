import numpy as np
import pdb
import unittest
import random

import eugene as eu
import isFunction as isF

#class TestFlagShelfWithFloats(unittest.TestCase)

#what if?
#        testShelves = {0: [[1,3]], 1: [1,3], 2: [], 3: []}
#                                            None^  None^
#still a function.


#what if?
#        testShelves = {0: [[1,3]], 1: [[0,4]], 2: [], 3: []}
#                                            None^  None^
#NOT a function.


class TestFlagShelfIntegers(unittest.TestCase):
    """
    Test FlagShelf's ability to correctly place flags in expected locations
    with expected values.

    Test Methods:
    test_flagShelfIntegers_1
    test_flagShelfIntegers_2
    test_flagShelfIntegers_3
    test_flagShelfIntegers_4
    """
    #simpleF1 HAS flags on bottom & IS a function.
    simpleF1 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
            [np.array([1,   2,  3,  2, 1]), 
             np.array([5,   6,  7,  6, 5]),
             np.array([10, 11, 12, 11, 10]),
             np.array([15, 16, 17, 16, 15])])
    #simpleF2 HAS NO flags & IS a function.
    simpleF2 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
            [np.array([2, 4, 6,   8, 10]), 
             np.array([3, 5, 7,   9, 11]),
             np.array([4, 6, 8,  10, 14]),
             np.array([5, 7, 9,  11, 15])])
    #simpleN1 HAS flags on bottom & IS NOT  function.
    simpleN1 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
            [np.array([1,   2,  3,  2, 1]), 
             np.array([5,   6,  7,  8, 9]),
             np.array([10, 11, 12, 18, 4]),
             np.array([15, 16, 17, 1,  5])])
    #simpleN2 HAS NO flags and IS NOT? a function.
    simpleN2 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
              [np.array([1,   2,  3,  4, 5]), 
               np.array([5,   6,  7,  8, 5]),
               np.array([10, 11, 12, 18, 4]),
               np.array([15, 16, 17, 1,  5])])

    def test_flagShelfIntegers_1(self):
        flaggy = isF.FlagShelf(self.simpleF1)
        flaggy.fillEntireShelf()
        testShelves = {0: [[0, 4], [1, 3]], 1: [[0, 4], [1, 3]], 
                       2: [[0, 4], [1, 3]], 3: [[0, 4], [1, 3]]}
        self.assertEquals(flaggy._shelves, testShelves)

    def test_flagShelfIntegers_2(self):
        flaggy = isF.FlagShelf(self.simpleF2)
        flaggy.fillEntireShelf()
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(flaggy._shelves, testShelves)

    def test_flagShelfIntegers_3(self):
        flaggy = isF.FlagShelf(self.simpleN1)
        flaggy.fillEntireShelf()
        testShelves = {0: [[0, 4], [1, 3]], 1: [], 2: [], 3: []}
        self.assertEquals(flaggy._shelves, testShelves)

    def test_flagShelfIntegers_4(self):
        flaggy = isF.FlagShelf(self.simpleN2)
        flaggy.fillEntireShelf()
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(flaggy._shelves, testShelves)

class TestFlagShelfIntegersInitialization(unittest.TestCase):
    """
    Test inilization of a FlagShelf when data frames hold integers.

    Test Methods:
        test_initialialization_1
        test_initialialization_2
        test_initialialization_3
        test_initialialization_4

    """
    simpleF1 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
            [np.array([1,   2,  3,  2, 1]), 
             np.array([5,   6,  7,  6, 5]),
             np.array([10, 11, 12, 11, 10]),
             np.array([15, 16, 17, 16, 15])])
    simpleF2 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
            [np.array([2, 4, 6,   8, 10]), 
             np.array([3, 5, 7,   9, 11]),
             np.array([4, 6, 8,  10, 14]),
             np.array([5, 7, 9,  11, 15])])
    simpleN1 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
            [np.array([1,   2,  3,  2, 1]), 
             np.array([5,   6,  7,  8, 9]),
             np.array([10, 11, 12, 18, 4]),
             np.array([15, 16, 17, 1,  5])])
    simpleN2 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
              [np.array([1,   2,  3,  4, 5]), 
               np.array([5,   6,  7,  8, 5]),
               np.array([10, 11, 12, 18, 4]),
               np.array([15, 16, 17, 1,  5])])
    
    def test_initialization_1(self):
        """
        Data frame is a function & has flags on the bottom shelf.
        """

        flaggy = isF.FlagShelf(self.simpleF1)
        #only fills bottom shelf during initialization.
        testShelves = {0: [[0,4], [1,3]], 1: [], 2: [], 3: []}
        self.assertEquals(testShelves, flaggy._shelves)

    def test_initialization_2(self):
        """
        Data frame is a function & has NO flags on bottom.
        """
        #'is a function' & 'has NO flags on bottom.'

        flaggy = isF.FlagShelf(self.simpleF2)
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_initialization_3(self):
        """
        Data frame is not a function & has flags on bottom.
        """
        #'is not a function' & 'has flags on bottom'

        flaggy = isF.FlagShelf(self.simpleN1)
        testShelves = {0: [[0,4], [1,3]], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_initialization_4(self):
        """
        Data frame is not a function & has NO flags on bottom.
        """
        #'is not a function' & 'has NO flags on bottom'

        flaggy = isF.FlagShelf(self.simpleN2)
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)





if __name__ == '__main__':
    unittest.main(exit=False)




