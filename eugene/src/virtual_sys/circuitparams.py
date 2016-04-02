# CircuitParams.py

""" Contains information corresponding to different
chaotic circuits, for use by VABCircuits. """

import numpy as np

             
Ai = np.array([[-2.017, 0, 0, 1, -1, 0, 0,lambda x:x**2],
              [-2.017, 0, 0, -1, -1, 0, 0, lambda x:x**2],
              [0, 0, -2.8, 0, 1, 1, 0, lambda x:x**2],
              [-.44, 0, -2, 0, 0, 1, -1, lambda x:x**2],
              [-.5, 0, -1, 0, 1, 1, 0, lambda x:x**2],
              [-0.3, 0., -0.3, 0., 0., -1., 1., lambda x: max(x, 0.)],
              [-0.3, 0., -0.3, 0., 0., -1., -1., lambda x: min(x, 0.)],             
              [-0.19, 0., -1., 0., -1., 2., 0., lambda x: np.tanh(x)],
              [-0.2, 0., -1., 0., 0., 1., 0., lambda x: np.sin(x)],
              [-0.2, 0., -1., 0., -1., 2.2, 0., lambda x: np.tanh(x)]
             ])
                      
initX = np.array([[0,0,1],
                   [0,0,-1],
                   [.5,-1,1],
                   [0,0,0],
                   [0,1,0],
                   [0., 0., 0.],
                   [0., 0., 0.],
                   [0., 1., 0.],
                   [0., 1., 0.],
                   [0., 1., 0.]
                   ])
                      
# now build a dictionary for VABCircuits to draw on

cdict = {} 
SprottList = range(len(Ai)) # set range as the number of circuits we have
for i in SprottList: 
    cdict[i] = [Ai[i,], initX[i,]]

