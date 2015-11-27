# CircuitParams.py

""" Contains information corresponding to different
chaotic circuits, for use by VABCircuits. """

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
                      
# now build a dictionary for VABCircuits to draw on

cdict = {} 
SprottList = range(4) # set range as the number of circuits we have
for i in SprottList: 
    cdict[i] = [Ai[i,], initX[i,]]


"""
Something I was trying before. 

class Ai_Row(object):
    
    def __init__(self,cr):
        self.cr = cr
        self._Ais = Ai[cr,]
        
class initX_Row(object):
    
    def __init__(self,cr):        
        self.cr = cr
        self.initXs = initX[cr,]

cdict = {}

SprottList = range(1,5)         where # stop = # of circuits + 1

for i in SprottList:
    cdict[i] = [Ai_Row(i), initX_Row(i)]

print cdict

the problem with that is that it was returning an array of an array"""

