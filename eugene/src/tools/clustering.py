# file: clustering.py

import pdb
import numpy as np
import copy

""" Provides a variety of methods for clustering on the basis of dynamical
distance matrices.
"""

def matrix_to_orderings(distance_matrix, row_labels=None, epsilon=0.):
    """ Converts a distance matrix into a list of ordered lists.
        distance_matrix: a 2D numpy array
    """
    dm = distance_matrix
    nn = dm.shape[0]

    if row_labels is None:
        row_labels = []
        for ii in range(nn):
            row_labels.append('r' + str(ii))
    elif len(row_labels) != nn:
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
                # strick ranking
                if epsilon == 0.:
                    tmp = []
                    for rank in row_ranking:
                        tmp.append(row_labels[rank])
                    rankings.append(tmp)
                # soft ranking
                else:
                    tmp = []
                    for kk in range(len(row_ranking)):
                        if kk ==0:
                            tmp.append([row_labels[row_ranking[kk]]])

                        elif dm[ii, row_ranking[kk]] > dm[ii, row_ranking[kk-1]] + epsilon:
                            for it in tmp:
                                it.append(row_labels[row_ranking[kk]])

                        else:
                            tmp2 = copy.deepcopy(tmp)
                            for it in tmp:
                                it.append(row_labels[row_ranking[kk-1]])
                            for it2 in tmp2:
                                it2[-1] = row_labels[row_ranking[kk]]
                                it2.append(row_labels[row_ranking[kk]])
                            tmp.extend(tmp2)
                    rankings.extend(tmp)

            else:
                old_row_ranking = copy.deepcopy(row_ranking)

    return(rankings)


def combinations(items, size):
    """ Input:
        items : a list of DISTINCT items from which to extact combinations of size = size
        size : how many items to choose
        Output:
        combos : a list of tuples consistuting the set of all combinations of size = size
    """
#    pdb.set_trace()

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
#                tmp = [it, completion]
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
    

def check_if_cluster(candidate, ordered_lists):
    """ The condition can be expressed as follows: For any distance ranking (ordered
    list) that begins with a member of the candidate cluster, each successive
    item in the ranking must belong to the candidate set until all members have
    been accounted for.
    """
    is_cluster = True
    for ol in ordered_lists:
        if ol[0] in candidate: 
        # ignore unless the ranking starts with a member
        # the candidate set
            switch = 0
            # run through the whole ranking
            for ii in range(1, len(ol)):
                if (((not ol[ii] in candidate) and (ol[ii-1] in candidate)) or
                    ((ol[ii] in candidate) and (not ol[ii-1] in candidate))):
                    switch += 1
            if switch > 1:
                is_cluster = False 

    return is_cluster


def qualitative_cluster(point_list, ordered_lists):
    """ Finds clusters of systems for which each member is more similar to one
    another than to any system outside the cluster on the basis of purely
    qualitative orderings.
    """
    clusters_found = []

    # loop over cluster sizes
    for size in range(2, len(point_list)):
        # loop over potential clusters
        potential_clusters = combinations(point_list, size)
        for cluster in potential_clusters:
            # determine whether this is a genuine cluster with respect to the
            # given set of ordered_lists
            if check_if_cluster(cluster, ordered_lists):
                clusters_found.append(cluster)
    
    return clusters_found
