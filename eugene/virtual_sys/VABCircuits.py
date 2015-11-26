#VABCircuits.py

"""
Simulates an electrical circuit with chaotic features, pulling the 
information for a given circuit from circuitparams. The jerk equation is 
expressed as a system of ODE's that is passed to scipy's odeint integrator."""
 

import circuitparams
import numpy as np
from scipy.integrate import odeint                 
from numpy import linspace    


class JerkCircuit(object):
    # Provides data corresponding to a particular circuit.
    cdict = circuitparams.cdict  
    def __init__(self, rowNum):
        self._Ais = self.cdict[rowNum][0]
        self._initXs = self.cdict[rowNum][1]
            
        try:   
            print(type(self._Ais))
            print(type(self._initXs))
            assert type(self._Ais) == np.ndarray and type(self._initXs) == np.ndarray
                                          
        
            def get_Ais(self):
                return self._Ais
                
            
            def get_initXs(self):
                return self._initXs
                
        except type(self._Ais) != np.ndarray or \
               type(self._initXs) != np.ndarray:
            
            print "UNACCEPTABLE. ONE MILLION YEARS DUNGEON."


    #now set up deriv to pass to odeint  
    
    def deriv(self,x,t0):
        # deriv sets up the following relationship: 
        # an array x, representing the system of ODE's, and 
        # some starting time t0 for which to return the derivatives
        Ai = self._Ais
        phi = Ai[7]
        return np.array([x[1], x[2], Ai[0] * x[2] + Ai[1] * phi(x[2]) \
            + Ai[2] * x[1] + Ai[3] * phi(x[1]) + Ai[4] * x[0] \
            + Ai[5] * phi(x[0]) + Ai[6]])
            
            
    def JerkSolver(self):  
        # returns an array containing the values for x, xdot, xdoot at 
        #  the times in t, beginning with their initial conditions      
        t0 = 0
        t = linspace(t0, 60, 10)
        x = odeint(self.deriv,self._initXs,t)
        return x

# just for demo     
Wilson = JerkCircuit(0)       
Vanessa = Wilson.JerkSolver()
print Vanessa