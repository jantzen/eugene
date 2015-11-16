#VABCircuits.py

#WIP, am struggling with how classes work so any feedback massively appreciated.

import numpy as np
from numpy import linspace



class CircuitRow(object):
    # Provides data corresponding to a particular circuit. 
    def __init__(self,circ):
        
        # build a dictionary of circuit variables
        self.circ = circ
        
        Ais = {1:[-2.017, 0, 0, 1, -1, 0, 0, lambda x:x**2], 
                       2:[-2.017, 0, 0, -1, -1, 0, 0, lambda x:x**2],
                       3:[0, 0, -2.8, 0, 1, 1, 0, lambda x:x**2],
                       4:[-.44, 0, -2, 0, 0, 1, -1, lambda x:x**2]}
               
 
        init_xs = {1: [0,0,1],
                  2: [0,0,-1],
                  3: [.5,-1,1],
                  4: [0,0,0]}
        
        # just checking the dictionary
        Ai = Ais.get(1)  
        print Ai
        print Ai[0]              
        
    def getinit_x(self):
        circ = self.circ 
        init_xs = self._init_xs
        return init_xs.get(circ)
        
    
    def getAis(self):
        circ = self.circ
        Ais = self._Ais      
        return Ais.get(circ)

print vars(CircuitRow)  #just checking

  
#now set up deriv to pass to odeint  
    
def deriv(x,t):
    # returns derivatives of the array x
    circ = CircuitRow.circ     
    Ai = CircuitRow.getAis(circ)
    phi = Ai[7]
    return np.array([x[1], x[2], Ai[0] * x[2] + Ai[1] * phi(x[2]) \
            + Ai[2] * x[1] + Ai[3] * phi(x[1]) + Ai[4] * x[0] \
            + Ai[5] * phi(x[0]) + Ai[6]])

class JerkFunctions(object):
# Produces a function based on the information for a particular CircuitRow.

    def __init__(self, circ):
        self.circ = circ        
        circ = CircuitRow(circ)    
    
    def updatevoltage(self):   
        from scipy.integrate import odeint                 
        circ = JerkFunctions.circ
        init_x = CircuitRow.getinit_x(circ)
        init_t = 0
        time = linspace(init_t, 60, 100)
        v = odeint(deriv,init_x,time)
        return v
        
    
print vars(JerkFunctions) #checking