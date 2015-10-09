# import scipy.integrate
from scipy.integrate import odeint
import time
import numpy as np


def deriv(x,curr_time):
    # returns derivatives in a list/array, is passed to the odeint function
    # x is a list or array of x-values, curr_time is the current time
    a = -2.017
    # get current time
    curr_time = time.time()
    return np.array([ x[1], x[2], a * x[2] + (x[1]) ** 2 - x[0] ])
    
# set up the other two arguments for ODEint. 

# first give the times 

times = np.linspace(0.0, 10.0, 100000) 

# then specify your initial conditions. 

xinit = np.array([0.0, 0.0, 1.0]) 

solution = odeint(deriv, xinit, times)

print solution

