# file: dmat_to_color_map.py
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from sklearn.manifold import MDS


def dmat_to_color_map(dmat, embedding_dim=3, map_dims=None, order='C', free_cores=2):
    """ Converts a distance matrix to a colored map of parameter space. To do
    so, the distance matrix is mapped to a set of coordinates in n-dimensional
    space (where n = embedding_dim) using the multidimensional scaling (MDS) 
    method in sklearn. The resulting embedding coordinates are scaled to fall 
    within 0 to 1, which can then be interpreted as colors.

    Inputs:
	dmat: an m x m symmetric matrix representing the dynamical distance between m points in parameter space.
	embedding_dim: the number of dimensions in which to embed the points (should be between 1 and 3)
	map_dims: user-specified dimensions, (a, b), for the output array where a * b = m
	free_cores: number of process threads (cores) to leave unengaged by the method

    Output:
	cspace_map: an a x b x embedding_dim array providing a color value for each pixel or cell 
	    in the a x b array of points

    """

    # verify dmat is square
    if not dmat.shape[0] == dmat.shape[1]:
        raise ValueError("dmat must be a square matrix")

    # if dimensions are given for the final map, verify they are acceptable
    if not map_dims is None:
        if not map_dims[0] * map_dims[1] == dmat.shape[0]:
            raise ValueError(
                    "cannot reshape {} pixels as a {} x {} array".format(
                        dmat.shape[0], map_dims[0], map_dims[1]))

    if map_dims is None:
        # is the width of dmat is a perfect square, assume a square array
        tmp = np.sqrt(dmat.shape[0])
        if tmp**2 == dmat.shape[0]:
            map_dims = (int(tmp), int(tmp))
        # otherwise, ask for dimensions
        else:
            raise ValueError("Cannot determine dimensions of coordinate map. Please provide a value for the map_dims arguments.")

    # compute the MDS embedding
    cpus = max(multiprocessing.cpu_count() - free_cores, 1)
    m = MDS(n_components=embedding_dim, dissimilarity='precomputed', n_jobs=cpus)
    reduced_coords = m.fit_transform(dmat)

    # scale to interpret as colors
    reduced_coords -= reduced_coords.min()
    reduced_coords *= 1.0 / reduced_coords.max()

    # build a matrix of color-values corresponding to parameter space
    cspace_map = reduced_coords.reshape(map_dims[0], map_dims[1],
            embedding_dim, order=order)

    return cspace_map
        
