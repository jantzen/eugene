# file: clustering.py

import pdb
import numpy as np
import copy

""" Provides a variety of methods for clustering on the basis of dynamical
distance matrices.
"""

def matrix_to_orderings(distance_matrix, row_labels=None):
    """ Converts a distance matrix into a list of ordered lists.
        distance_matrix: a 2D numpy array
    """
    dm = distance_matrix
    nn = dm.shape[0]

    if row_labels is None:
        row_labels = []
        for ii in range(nn):
            row_labels.append('r' + str(ii))
    elif len(row_labels != nn):
        raise ValueError("Row labels list length does not match distance matrix shape.")

    rankings = []
    for ii in range(nn):
        # bubble sort the row
        row_ranking = list(range(nn))
        old_row_ranking = list(range(nn))
        keep_sorting = True
        while keep_sorting:
            for jj in range(len(row_ranking)-1):
                if dm[ii, row_ranking[jj]] > dm[ii, row_ranking[jj+1]]:
                    # swap elements in the ranking
                    tmp = row_ranking[jj+1]
                    row_ranking[jj+1] = row_ranking[jj]
                    row_ranking[jj] = tmp
            if row_ranking == old_row_ranking:
                keep_sorting = False
                # convert ranks to labels
                tmp = []
                for rank in row_ranking:
                    tmp.append(row_labels[rank])
                rankings.append(tmp)
            else:
                old_row_ranking = row_ranking

    return(rankings)


def combinations(items, size):
    """ Input:
        items : a list of DISTINCT items from which to extact combinations of size = size
        size : how many items to choose
        Output:
        combos : a list of lists consistuting the set of all combinations of size = size
    """
#    pdb.set_trace()

    if size > len(items):
        raise ValueError("Number of items chosen cannot exceed total number of items.")

    # base case
    if size == 1:
        return items

    # recursive step
    else:
        combos = []
        for it in items:
            remaining_items = copy.deepcopy(items)
            remaining_items.remove(it)
            for completion in combinations(remaining_items, size-1):
                tmp = [it, completion]
                tmp.sort()
                combos.append(tmp)
        # remove the duplicates
        dd = set()
        for co in combos:
            dd.add(tuple(co))
        combos = list(dd)

    return combos
    


def qualitative_cluster(point_list, ordered_lists):
    """ Finds clusters of systems for which each member is more similar to one
    another than to any system outside the cluster on the basis of purely
    qualitative orderings.
    """
    # determine the largest possible cluster size of interest
    max_cluster_size = len(point_list) - 1

    potential_clusters = dict([])

    # loop over cluster sizes
    for size in range(max_cluster_size):
        potential_clusters[size] = combinations(point_list, size)



