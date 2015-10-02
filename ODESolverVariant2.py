import scipy.integrate
from scipy.integrate import odeint
import numpy
from numpy import linspace 


def deriv(x,t):
# return derivatives of the array x
    a = -2.017
    return numpy.array([ x[1], x[2], a * x[2] + (x[1]) ** 2 - x[0] ])

# set up variables for odeint function

# first give the timespan - should virtual time be here?  

time = linspace(0.0, 10.0, 100000) 

# then specify your initial conditions. 

xinit = numpy.array([0.0, 0.0, 1.0]) 

x = odeint(deriv, xinit, time)

print x


