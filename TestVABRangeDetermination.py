#TestVABRangeDetermination.py
#Determining a range is fairly subjective, so this really functions
#more as a trial, than a test, but this can be used to test the helper functions
#as well.

import VABRangeDetermination as V
import numpy as np

a = np.array([2,3,4,5,5.5,6,6.5,7,6.5,6,5.5,5,4,3,2,1.5,1,1,1.5,1])
print V.findRange(a)