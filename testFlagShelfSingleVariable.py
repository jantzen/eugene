import numpy as np
import pdb
import unittest

import eugene as eu
import isFunction as isF

# class TestFlagShelfFloatsInitialization(unittest.TestCase):
    # def test_initialization(self):
    #     #test for noisy case.
    #     err = 0.7
    #     noisyDF = simpleDF
    #     # pdb.set_trace()
    #     for tv in range(len(noisyDF._target_values)):
    #         for v in range(tv):
    #             noise = random.uniform(-err, err)
    #             noisyDF._target_values[tv][v] += noise
    #     flaggy = isF.FlagShelf(simpleDF)
    #     #only bottom shelf is filled during initialization.
    #     testFlaggy = {0: [[0,4], [1,3]], 1: [[0,4], [1,3]], 
    #                   2: [],             3: []}
    #     self.assertEquals(testFlaggy, flaggy._shelves)


class TestFlagShelfIntegersFirstShelf(unittest.TestCase):
    """
    Test filling the first shelf of a FlagShelf when data
    frames hold integers.

    --------------ATTRIBUTES-------------------------
    Test Methods:
        test_firstShelf_1
        test_firstShelf_2
        test_firstShelf_3
        test_firstShelf_4

    """
    
    def test_firstShelf_1(self):
        """
        Data frame is a function & has flags on the bottom shelf.
        """
        simpleF1 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
                                          [np.array([1,   2,  3,  2, 1]), 
                                           np.array([5,   6,  7,  6, 5]),
                                           np.array([10, 11, 12, 11, 10]),
                                           np.array([15, 16, 17, 16, 15])])

        flaggy = isF.FlagShelf(simpleF1)
        testShelves = {0: [[0,4], [1,3]], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)
        # pdb.set_trace()
        flaggy.fillShelf(1)
        testShelves[1] = [[0, 4], [1, 3]]
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_firstShelf_2(self):
        """
        Data frame is a function & has NO flags on bottom.
        """
        simpleF2 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
                                          [np.array([2, 4, 6,   8, 10]), 
                                           np.array([3, 7, 9,  11, 13]),
                                           np.array([4, 8, 10, 12, 14]),
                                           np.array([5, 9, 11, 13, 15])])

        flaggy = isF.FlagShelf(simpleF2)
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_firstShelf_3(self):
        """
        Data frame is not a function & has flags on bottom.
        """
        simpleN1 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
                            [np.array([1,   2,  3,  2, 1]), 
                             np.array([5,   6,  7,  8, 9]),
                             np.array([10, 11, 12, 18, 4]),
                             np.array([15, 16, 17, 1, 5])])

        flaggy = isF.FlagShelf(simpleN1)
        testShelves = {0: [[0,4], [1,3]], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_firstShelf_4(self):
        """
        Data frame is not a function & has NO flags on bottom.
        """
        simpleN2 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
                            [np.array([1,   2,  3,  4, 5]), 
                             np.array([5,   6,  7,  8, 5]),
                             np.array([10, 11, 12, 18, 4]),
                             np.array([15, 16, 17, 1, 5])])

        flaggy = isF.FlagShelf(simpleN2)
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)




class TestFlagShelfIntegersInitialization(unittest.TestCase):
    """
    Test inilization of a FlagShelf when data frames hold integers.

    --------------ATTRIBUTES-------------------------
    Test Methods:
        test_initialialization_1
        test_initialialization_2
        test_initialialization_3
        test_initialialization_4

    """
    
    def test_initialialization_1(self):
        #'is a function' & 'has flags on bototm'
        simpleF1 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
                                          [np.array([1,   2,  3,  2, 1]), 
                                           np.array([5,   6,  7,  6, 5]),
                                           np.array([10, 11, 12, 11, 10]),
                                           np.array([15, 16, 17, 16, 15])])

        flaggy = isF.FlagShelf(simpleF1)
        #only fills bottom shelf during initialization.
        testShelves = {0: [[0,4], [1,3]], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_initialialization_2(self):
        #'is a function' & 'has NO flags on bottom.'
        simpleF2 = eu.interface.DataFrame(1, np.array([0,   1,  2,  3, 4]), 2,
                                          [np.array([2, 4, 6,   8, 10]), 
                                           np.array([3, 7, 9,  11, 13]),
                                           np.array([4, 8, 10, 12, 14]),
                                           np.array([5, 9, 11, 13, 15])])

        flaggy = isF.FlagShelf(simpleF2)
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_initialialization_3(self):
        #'is not a function' & 'has flags on bottom'
        simpleN1 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
                            [np.array([1,   2,  3,  2, 1]), 
                             np.array([5,   6,  7,  8, 9]),
                             np.array([10, 11, 12, 18, 4]),
                             np.array([15, 16, 17, 1, 5])])

        flaggy = isF.FlagShelf(simpleN1)
        testShelves = {0: [[0,4], [1,3]], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)

    def test_initialialization_4(self):
        #'is not a function' & 'has NO flags on bottom'
        simpleN2 = eu.interface.DataFrame(1, np.array([0, 1, 2, 3, 4]), 2, 
                            [np.array([1,   2,  3,  4, 5]), 
                             np.array([5,   6,  7,  8, 5]),
                             np.array([10, 11, 12, 18, 4]),
                             np.array([15, 16, 17, 1, 5])])

        flaggy = isF.FlagShelf(simpleN2)
        testShelves = {0: [], 1: [], 2: [], 3: []}
        self.assertEquals(type(testShelves), type(flaggy._shelves))
        self.assertEquals(testShelves, flaggy._shelves)



#class TestFlagShelfWithFloats(unittest.TestCase)

if __name__ == '__main__':
    unittest.main(exit=False)




