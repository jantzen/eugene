# test_robustness.py

import numpy as np
from eugene.src.tools.LVDSim import *
from eugene.src.tools.StochasticTools import *
import unittest

class TestRobustness(unittest.TestCase):

    def test_same_more_data(self):
        r1 = np.array([1.,2.])
        r2 = 1.5 * r1
    
        k1 = k2 = np.array([100.,100.,100.])
    
        alpha1 = np.array([[1.,0.5],[0.7,1.]])
        alpha2 = alpha1
    
        init1 = np.array([0.5,0.5])
        init2 = init1
    
        init_trans1 = np.array([0.8,0.8])
        init_trans2 = init_trans1
    
        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2] 
    
        overlay = lambda x: np.mean(x, axis=1)
    
        data = simData([params1, params2], 5., 500, overlay)
    
        ddata = downSample(data)
    
        blocks = tuplesToBlocks(ddata)
    
        kdes = blocksToKDEs(blocks)
    
        densities = KDEsToDensities(kdes)
    
        dmat = distanceH2D(densities)
        
        dist1 = dmat[0][1] - dmat[0][0]
        
        more_data = simData([params1, params2], 5., 1000, overlay)
        
        more_ddata = downSample(more_data)
        
        more_blocks = tuplesToBlocks(more_ddata)
        
        more_kdes = blocksToKDEs(more_blocks)
        
        more_densities = KDEsToDensities(more_kdes)
        
        dmat2 = distanceH2D(more_densities)
        
        dist2 = dmat2[0][1] - dmat[0][0]
        
        self.assertTrue(dist1 > dist2)
        
        
    def test_same_more_info(self):
        r1 = np.array([1.,2.])
        r2 = 1.5 * r1
    
        k1 = k2 = np.array([100.,100.,100.])
    
        alpha1 = np.array([[1.,0.5],[0.7,1.]])
        alpha2 = alpha1
    
        init1 = np.array([0.5,0.5])
        init2 = init1
    
        init_trans1 = np.array([0.8,0.8])
        init_trans2 = init_trans1
    
        params1 = [r1, k1, alpha1, init1, init_trans1]
        params2 = [r2, k2, alpha2, init2, init_trans2] 
    
        overlay = lambda x: np.mean(x, axis=1)
    
        data = simData([params1, params2], 5., 500, overlay)
    
        ddata = downSample(data)
    
        blocks = tuplesToBlocks(ddata)
    
        kdes = blocksToKDEs(blocks)
    
        densities = KDEsToDensities(kdes)
    
        dmat = distanceH2D(densities)
        
        dist1 = dmat[0][1] - dmat[0][0]
        
        less_overlay = lambda x: x
        
        more_data = simData([params1, params2], 5., 500, less_overlay)
        
        more_ddata = downSample(more_data)
        
        more_blocks = tuplesToBlocks(more_ddata)
        
        more_kdes = blocksToKDEs(more_blocks)
        
        more_densities = KDEsToDensities(more_kdes)
        
        dmat2 = distanceH2D(more_densities)
        
        dist2 = dmat2[0][1] - dmat[0][0]
        
        self.assertTrue(dist1 > dist2)
        


if __name__ == '__main__':
    unittest.main()