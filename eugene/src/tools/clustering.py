# file: clustering.py

import pdb
import numpy as np
import copy

""" Provides a variety of methods for clustering on the basis of dynamical
distance matrices.
"""

class DistanceMatrix( object ):
    """ Provides for distance matrix objects that collect matrices, their
    associated row labels, and methds for extracting matrix entries from ro
    labels.
    """
    def __init__(self, matrix, labels, sort_labels=True):
        """ Input:
                matrix : A 2d numpy array representing an n x n distance matrix
                labels : A list of labels for each row of the matrix. By
                default, this is sorted.
        """
        self.matrix = matrix
        if len(set(labels)) < len(labels):
            raise ValueError("Labels must be unique.")

        if not len(labels) == matrix.shape[0]:
            raise ValueError("Number of labels must match matrix dimensions.")

        if sort_labels:
            self.labels = labels
            self.labels.sort()
        else:
            self.labels = labels

    def distance(self, label1, label2):
        if not (label1 in self.labels and label2 in self.labels):
            raise ValueError("Given row labels not associated with this distance matrix.")
        index1 = self.labels.index(label1)
        index2 = self.labels.index(label2)
        return self.matrix[index1, index2]


def combinations(items, size):
    """ Input:
        items : a list of DISTINCT items from which to extact combinations of size = size
        size : how many items to choose
        Output:
        combos : a list of tuples consistuting the set of all combinations of size = size
    """

    if size > len(items):
        raise ValueError("Number of items chosen cannot exceed total number of items.")

    # base case
    if size == 1:
        tmp = []
        for it in items:
            tmp.append((it,))
        return tmp

    # recursive step
    else:
        combos = []
        for it in items:
            remaining_items = copy.deepcopy(items)
            remaining_items.remove(it)
            for completion in combinations(remaining_items, size-1):
                tmp = [it]
                tmp.extend(list(completion))
                tmp.sort()
                combos.append(tmp)
        # remove the duplicates
        dd = set()
        for co in combos:
            dd.add(tuple(co))
        combos = list(dd)

    return combos
    

def find_greatest_separation(distance_matrix, candidate_cluster):
    greatest_sep = 0.
    for element1 in candidate_cluster:
        for element2 in candidate_cluster:
            if (element1 in distance_matrix.labels and 
                    element2 in distance_matrix.labels):
                dist = distance_matrix.distance(element1, element2)
                if dist > greatest_sep:
                    greatest_sep = dist
    return greatest_sep


def check_if_cluster(candidate, distance_matrices, labels, epsilon=0.,
        relative_epsilon=False):
    """ 
    Input:
        candidate : a list or tuple of labels constituting the putative cluster
        distance_matrices : a list of DistanceMatrix objects with labels from labels
        labels : a list of tags for each system in the entire set considered
    """
    is_cluster = True
    candidate = set(candidate)
    complement = set(labels) - candidate
#    # find the greatest separation between elements of the candidate cluster
#    gs = 0.
#    for dm in distance_matrices:
#        tmp = find_greatest_separation(dm, candidate)
#        if tmp > gs:
#            gs = tmp
    # check whether each element of the complement is more than gs + epsilon away from
    # every element of the candidate _in every distance matrix_
    for dm in distance_matrices:
        gs = find_greatest_separation(dm, candidate)
        if relative_epsilon:
            epsilon = epsilon * np.std(dm.matrix)
        for label1 in complement:
            for label2 in candidate:
                if (label1 in dm.labels and 
                        label2 in dm.labels and
                        dm.distance(label1, label2) < gs + epsilon):
                    is_cluster = False
    return is_cluster


def qualitative_cluster(distance_matrices, labels, epsilon=0., relative_epsilon=False):
    """ Finds clusters of systems for which each member is more similar to one
    another than to any system outside the cluster on the basis of purely
    qualitative orderings.
    """
    clusters_found = []

    # loop over cluster sizes
    for size in range(2, len(labels)):
        pdb.set_trace()
        # loop over potential clusters
        potential_clusters = combinations(labels, size)
        for cluster in potential_clusters:
            # determine whether this is a genuine cluster with respect to the
            # given set of ordered_lists
            if check_if_cluster(cluster, distance_matrices, labels,
                    epsilon=epsilon, relative_epsilon=relative_epsilon):
                clusters_found.append(cluster)
    
    return clusters_found
