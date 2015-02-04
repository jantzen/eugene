from __future__ import division
from sympy import *
import math

x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

def transformOfEvolve(f, g, t):
    #evolve then transform
    return lambda x: f(g(x, t))
    
def evolveOfTransform(g, f, t):
    #transform then evolve
    return lambda x: g(f(x), t)
    
transform = lambda x: 2*x

evolution = lambda x,t: x-.5*9.81*pow(t,2)
    #falling due to gravity: x-.5*9.81*pow(t,2)
    #radioactive decay constant of 5: x*pow(math.e,-5/t)

print transform(evolution(8, 1))
print evolution(transform(8), 1)
toe = transformOfEvolve(transform, evolution, 1)
eot = evolveOfTransform(evolution, transform, 1)
print toe(8)
print eot(8)
eq = toe(x)-eot(x)
print simplify(eq)