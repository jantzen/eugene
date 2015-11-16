#VABCircuits.py

#WIP, am struggling with how classes work so any feedback massively appreciated.

import numpy as np
from numpy import linspace



class CircuitRow(object):
    # Provides data corresponding to a particular circuit. 
    def __init__(self,circ):
        
        # build a dictionary of Ais
        self.circ = circ
        
        Ai = {1:[-2.017, 0, 0, 1, -1, 0, 0], 
              2:[-2.017, 0, 0, -1, -1, 0, 0],
              3:[0, 0, -2.8, 0, 1, 1, 0],
              4:[-.44, 0, -2, 0, 0, 1, -1]}


        phi = {1: lambda x:x**2,
               2: lambda x:x**2,
               3: lambda x:x**2,
               4: lambda x:x**2}
               
 
        init_x = {1: [0,0,1],
                  2: [0,0,-1],
                  3: [.5,-1,1],
                  4: [0,0,0]}
                  
        
    def getinit_x(self):
        circ = CircuitRow.circ
        init_xs = CircuitRow._init_x
        init_x = init_xs.get(circ)
        return init_x
    
    def getAi(self):
        circ = CircuitRow.circ
        Ais = CircuitRow._Ai       
        Ai = Ais.get(circ)
        return Ai
 
    def getphi(self):
        circ = CircuitRow.circ
        phis = CircuitRow._phi
        phi = phis.get(circ)
        return phi
  
        
    #now set up deriv to pass to odeint  
    
def deriv(x,t):
        # returns derivatives of the array x
    circ = CircuitRow.circ     
    Ai = CircuitRow.getAi(circ)
    phi = CircuitRow.getphi(circ)
    return np.array([x[1], x[2], Ai[0] * x[2] + Ai[1] * phi(x[2]) \
            + Ai[2] * x[1] + Ai[3] * phi(x[1]) + Ai[4] * x[0] \
            + Ai[5] * phi(x[0]) + Ai[6]])


class JerkFunctions(object):
# Produces a function based on the information for a particular CircuitRow.

    def __init__(self, circ):
        circ = CircuitRow(circ)    
    
    def updatevoltage(self):   
        from scipy.integrate import odeint                 
        circ = JerkFunctions.circ
        init_x = CircuitRow.getinit_x(circ)
        init_t = 0
        time = linspace(init_t, 60, 100)
        v = odeint(deriv,init_x,time)
        return v
        print v
        

