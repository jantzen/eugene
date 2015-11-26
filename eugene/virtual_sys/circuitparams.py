# CircuitParams.py

""" Contains information corresponding to different
chaotic circuits, to be pulled by VABCircuits. """

import numpy as np


             
Ai = np.array([[-2.017, 0, 0, 1, -1, 0, 0,lambda x:x**2],
              [-2.017, 0, 0, -1, -1, 0, 0, lambda x:x**2],
              [0, 0, -2.8, 0, 1, 1, 0, lambda x:x**2],
              [-.44, 0, -2, 0, 0, 1, -1, lambda x:x**2]
              ])
                      
initX = np.array([[0,0,1],
                   [0,0,-1],
                   [.5,-1,1],
                   [0,0,0]])
                      


class Ai_Row(object):
    
    def __init__(self,cr):
        self.cr = cr
        self._Ais = Ai[cr,]
        
class initX_Row(object):
    
    def __init__(self,cr):        
        self.cr = cr
        self.initXs = initX[cr,]


# now build a dictionary of circuit objects

# here's one way we could do it

cdict = {}

cdict[0] = [Ai[0,], initX[0,]]
cdict[1] = [Ai[1,], initX[1,]]
cdict[2] = [Ai[2,], initX[2,]]
cdict[3] = [Ai[3,], initX[3,]]
print cdict


"""but I like the following too:

cdict = {}

SprottList = range(1,5)         where # stop = # of circuits + 1

for i in SprottList:
    cdict[i] = [Ai_Row(i), initX_Row(i)]

print cdict"""

