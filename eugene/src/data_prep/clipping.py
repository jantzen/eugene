# File: clipping.py

import numpy as np
from eugene.src.auxiliary.probability import EnergyDistance
import copy

#Classes:
#   none

#Functions:
#   clip_segments_1d
#   clip_segments
#   clip_to_match_1d
#   clip_to_match
#   zip_curves_1d
#   zip_curves


###################################################################
###################################################################
#Functions


def clip_segments_1d(A, B, minimum_length, steps=10):
    """ This covers the 1-d case of the more general function clip_segments. i
    Implements dynamical scaling: Clips either the curves in A or the curves
    in B in order to minimize the mean energy distance between each point in A
    and the corresponding point in B. Assumes that the curves in lists A and B are 
    all of the same length.

        Inputs:
            A: a list of 1-d np-arrays of length n
            B: a list of 1-d np-arrays of length n
            minimum_length: the shortest allowed array length after clipping 
                (0 < minimum_length <= n)
            steps: an integer indicating the number of steps to take in searching 
                for a cut point between index minimum_length and (n-1)

        Returns:
            A_clipped, B_clipped (lists of np-arrays where one set has 
            been cut at the optimal index)
    """

    if minimum_length > max(A[0].shape):
        minimum_length = max(A[0].shape)
    elif minimum_length <= 0:
            minimum_length =1
            raise Warning("minimum_length set to 1")

    # Build lists of segments at a time slice
    A_len = max(A[0].shape)
    A_slices = [[] for i in range(A_len)]
    for ii in range(A_len):
        for segment in A:
            segment = np.squeeze(segment)
            A_slices[ii].append(segment[ii])
        A_slices[ii] = np.array(A_slices[ii])
    B_len = max(B[0].shape)
    B_slices = [[] for i in range(B_len)]
    for ii in range(B_len):
        for segment in B:
            segment = np.squeeze(segment)
            B_slices[ii].append(segment[ii])
        B_slices[ii] = np.array(B_slices[ii])

    # adjust number of steps and determine step size
    if steps > A_len - minimum_length:
        steps = A_len - minimum_length
    elif steps < 2:
        steps = 2
    step_size = A_len / steps

    # Find best point to clip A relative to B
    distances_A = []
    indices_A = []
    # loop over cut-points
    for index in range(minimum_length, A_len, int(step_size)):
        delta = float(B_len) / float(index+1)
        # align and compute average distance
        d = []
        for a in range(index):
            b = int(a * delta)
            tmp = EnergyDistance(A_slices[a].reshape(-1,1),
                    B_slices[b].reshape(-1,1))
            d.append(tmp)
        distances_A.append(np.mean(d))
        indices_A.append(index)
    mm = np.argmin(distances_A)
    best_distance_A = distances_A[mm]
    best_cut_index_A = indices_A[mm]

    # Find best point to clip B relative to A
    distances_B = []
    indices_B = []
    # loop over cut-points
    for index in range(minimum_length, B_len, int(step_size)):
        delta = float(A_len) / float(index+1)
        # align and compute average distance
        d = []
        for b in range(index):
            a = int(b * delta)
            tmp = EnergyDistance(B_slices[b].reshape(-1,1),
                    A_slices[a].reshape(-1,1))
            d.append(tmp)
        distances_B.append(np.mean(d))
        indices_B.append(index)
    mm = np.argmin(distances_B)
    best_distance_B = distances_B[mm]
    best_cut_index_B = indices_B[mm]

    # Choose the best set to clip, and return results
    if best_distance_A < best_distance_B:
        B_clipped = B
        A_clipped = []
        for segment in A:
            A_clipped.append(segment[:best_cut_index_A])
    else:
        A_clipped = A
        B_clipped = []
        for segment in B:
            B_clipped.append(segment[:best_cut_index_B])

    return A_clipped, B_clipped


def clip_segments(A, B, minimum_length, steps=10):
    """ Implements dynamical scaling: Clips either the curves in A or the curves
    in B in order to minimize the mean energy distance between each point in A
    and the corresponding point in B. Assumes that the curves in lists A and B are 
    all of the same length.

        Inputs:
            A: a list of np-arrays of length n
            B: a list of np-arrays of length n
            minimum_length: the shortest allowed array length after clipping 
                (0 < minimum_length <= n)
            steps: an integer indicating the number of steps to take in searching 
                for a cut point between index minimum_length and (n-1)

        Returns:
            A_clipped, B_clipped (lists of np-arrays where one set has 
            been cut at the optimal index)
    """

    if A[0].ndim == 1 or min(A[0].shape)==1:
        print("Clipping 1-D segments...")
        return clip_segments_1d(A, B, minimum_length, steps=steps)

    # Build lists of segments at a time slice
    A_len = A[0].shape[1]
    A_slices = [[] for i in range(A_len)]
    for ii in range(A_len):
        for segment in A:
            try:
                if not segment.shape[1] == A_len:
                    raise ValueError(
                        "All curves must be of the same length. {0} != {1}".format(
                            A_len, segment.shape[1]))
                A_slices[ii].append(segment[:,ii].reshape(1,-1))
            except ValueError as e:
                print(str(e))
                print("Dropping bad curve.")
        A_slices[ii] = np.concatenate(A_slices[ii], axis=0)
    B_len = B[0].shape[1]
    B_slices = [[] for i in range(B_len)]
    for ii in range(B_len):
        for segment in B:
            try:
                if not segment.shape[1] == B_len:
                    raise ValueError(
                        "All curves must be of the same length. {0} != {1}".format(
                            B_len, segment.shape[1]))
                B_slices[ii].append(segment[:,ii].reshape(1,-1))
            except ValueError as e:
                print(str(e))
                print("Dropping bad curve.")
        B_slices[ii] = np.concatenate(B_slices[ii], axis=0)

    # adjust number of steps and determine step size
    if steps > A_len - minimum_length:
        steps = A_len - minimum_length
    elif steps < 2:
        steps = 2
    step_size = A_len / steps

    # Find best point to clip A relative to B
    distances_A = []
    indices_A = []
    # loop over cut-points
    for index in range(minimum_length, A_len, int(step_size)):
        delta = float(B_len) / float(index+1)
        # align and compute average distance
        d = []
        for a in range(index):
            b = int(a * delta)
            tmp = EnergyDistance(A_slices[a],
                    B_slices[b])
            d.append(tmp)
        distances_A.append(np.mean(d))
        indices_A.append(index)
    mm = np.argmin(distances_A)
    best_distance_A = distances_A[mm]
    best_cut_index_A = indices_A[mm]

    # Find best point to clip B relative to A
    distances_B = []
    indices_B = []
    # loop over cut-points
    for index in range(minimum_length, B_len, int(step_size)):
        delta = float(A_len) / float(index+1)
        # align and compute average distance
        d = []
        for b in range(index):
            a = int(b * delta)
            tmp = EnergyDistance(B_slices[b],
                    A_slices[a])
            d.append(tmp)
        distances_B.append(np.mean(d))
        indices_B.append(index)
    mm = np.argmin(distances_B)
    best_distance_B = distances_B[mm]
    best_cut_index_B = indices_B[mm]

    # Choose the best set to clip, and return results
    if best_distance_A < best_distance_B:
        B_clipped = B
        A_clipped = []
        for segment in A:
            A_clipped.append(segment[:,:best_cut_index_A])
    else:
        A_clipped = A
        B_clipped = []
        for segment in B:
            B_clipped.append(segment[:,:best_cut_index_B])

    return A_clipped, B_clipped


def clip_to_match_1d(A_clipped, B_clipped, C, D):
    """ This is the 1-d companion to the more general function, clip_to_match. 
    Given four sets of curves, 

        Inputs:
            A_clipped: 
            B_clipped: 
            C:
            D:

        Returns:
            C_clipped, D_clipped:
    """


    cut_len1 = len(A_clipped[0])
    cut_len2 = len(B_clipped[0])

    if cut_len1 < cut_len2:
        D_clipped = copy.deepcopy(D)
        C_clipped = []
        for segment in C:
            C_clipped.append(segment[:cut_len1])
    elif cut_len1 > cut_len2:
        C_clipped = copy.deepcopy(C)
        D_clipped = []
        for segment in D:
            D_clipped.append(segment[:cut_len2])
    else:
        C_clipped = copy.deepcopy(C)
        D_clipped = copy.deepcopy(D)

    return C_clipped, D_clipped


def clip_to_match(A_clipped, B_clipped, C, D):

    if A_clipped[0].ndim == 1 or min(A_clipped[0].shape)==1:
        print("Clipping 1-D segments to match...")
        return clip_to_match_1d(A_clipped, B_clipped, C, D)


    cut_len1 = A_clipped[0].shape[1]
    cut_len2 = B_clipped[0].shape[1]

    if cut_len1 < cut_len2:
        D_clipped = copy.deepcopy(D)
        C_clipped = []
        for segment in C:
            C_clipped.append(segment[:,:cut_len1])
    elif cut_len1 > cut_len2:
        C_clipped = copy.deepcopy(C)
        D_clipped = []
        for segment in D:
            D_clipped.append(segment[:,:cut_len2])
    else:
        C_clipped = copy.deepcopy(C)
        D_clipped = copy.deepcopy(D)

    return C_clipped, D_clipped


def zip_curves_1d(A_clipped, B_clipped, C_clipped, D_clipped):

    data1 = []
    data2 = []

    for index, seg in enumerate(A_clipped):
        if len(seg) == len(C_clipped[index]):
            data1.append(np.vstack((seg.reshape(1,-1),
                C_clipped[index].reshape(1,-1))))
        else:
            s = np.min([len(seg), len(C_clipped[index])])
            data1.append(np.vstack((seg[:s].reshape(1,-1),
                C_clipped[index][:s].reshape(1,-1))))
    data1 = np.concatenate(data1, axis=1)
    
    for index, seg in enumerate(B_clipped):
        if len(seg) == len(D_clipped[index]):
            data2.append(np.vstack((seg.reshape(1,-1),
                D_clipped[index].reshape(1,-1))))
        else:
            s = np.min([len(seg), len(D_clipped[index])])
            data2.append(np.vstack((seg[:s].reshape(1,-1),
                D_clipped[index][:s].reshape(1,-1))))
    data2 = np.concatenate(data2, axis=1)

    return data1, data2


def zip_curves(A_clipped, B_clipped, C_clipped, D_clipped):

    if A_clipped[0].ndim == 1 or min(A_clipped[0].shape)==1:
        print("Zipping 1-D segments...")
        return zip_curves_1d(A_clipped, B_clipped, C_clipped, D_clipped)

    data1 = []
    data2 = []

    for index, seg in enumerate(A_clipped):
        if seg.shape[1] == C_clipped[index].shape[1]:
            data1.append(np.vstack((seg, C_clipped[index])))
        else:
            s = np.min([seg.shape[1], C_clipped[index].shape[1]])
            data1.append(np.vstack((seg[:,:s], C_clipped[index][:,:s])))
    data1 = np.concatenate(data1, axis=1)

    for index, seg in enumerate(B_clipped):
        if seg.shape[1] == D_clipped[index].shape[1]:
            data2.append(np.vstack((seg, D_clipped[index])))
        else:
            s = np.min([seg.shape[1], D_clipped[index].shape[1]])
            data2.append(np.vstack((seg[:,:s], D_clipped[index][:,:s])))
    data2 = np.concatenate(data2, axis=1)

    return data1, data2
