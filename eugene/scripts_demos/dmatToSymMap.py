# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:46:03 2018

@author: Colin
"""

import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np

def dmatToSymMap(dmat, shape, n=3, order='C'):
    m = MDS(n_components=n, dissimilarity='precomputed')
    reduced = m.fit_transform(dmat)
    reduced -= reduced.min()
    reduced *= 1.0/reduced.max()
    reduced = np.reshape(reduced, (shape[0], shape[1], 3), order=order)
    plt.figure()
    plt.imshow(reduced, interpolation='nearest', origin='lower')
    plt.show()