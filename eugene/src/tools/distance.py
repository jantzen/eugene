# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:06:01 2019

@author: Colin, Ben
"""

import numpy as np
import copy
from eugene.src.auxiliary.probability import EnergyDistance
import sys
sys.path.insert(0, '../../../AutomatedDiscovery/DADS2018/')
import clipping

def _distance(untrans, trans, min_length, clip=True):
    ii = 0
    jj = 1

    # untrans[system, rep, species, t] or untrans[system, rep, t]
    # untrans[ii][rep, species, t] or untrans[ii][rep, t]
    print(np.shape(untrans[ii]))
    condition1 = untrans[ii]
    condition2 = untrans[jj]

    if clip:
        # clip the data
        tmp_untrans1, tmp_untrans2 = clipping.clip_segments(condition1, condition2,
                min_length)
        cut_len1 = np.shape(tmp_untrans1[0])[-1]
        cut_len2 = np.shape(tmp_untrans2[0])[-1]
    else:
        tmp_untrans1 = condition1
        tmp_untrans2 = condition2
        cut_len1 = np.shape(condition1[0])[-1]
        cut_len2 = np.shape(condition2[0])[-1]

    # double check
    a = np.shape(condition1[0])[-1]
    b = np.shape(condition2[0])[-1]

    assert cut_len1 == a or cut_len2 == b

    # the tmp_trans1.append(segment[:cut_len1]) line won't work if segment is nd
    # so only take from the last axis
    if cut_len1 < cut_len2:
        tmp_trans2 = copy.deepcopy(trans[jj])
        tmp_trans1 = []
        for segment in trans[ii]:
            # tmp_trans1.append(segment[:cut_len1])
            tmp_trans1.append(np.take(segment, range(cut_len1), -1))
    elif cut_len1 > cut_len2:
        tmp_trans1 = copy.deepcopy(trans[ii])
        tmp_trans2 = []
        for segment in trans[jj]:
            # tmp_trans2.append(segment[:cut_len2])
            tmp_trans2.append(np.take(segment, range(cut_len2), -1))
    else:
        tmp_trans1 = copy.deepcopy(trans[ii])
        tmp_trans2 = copy.deepcopy(trans[jj])

    data1 = []
    data2 = []
    untrans1c = []
    untrans2c = []
    for index, seg in enumerate(tmp_untrans1):
        if seg.shape[-1] == tmp_trans1[index].shape[-1]:
            data1.append(np.vstack((seg, tmp_trans1[index])))
            untrans1c.append(seg)
        else:
            s = np.min([seg.shape[-1], tmp_trans1[index].shape[-1]])
            data1.append(np.vstack((seg[:,:s], tmp_trans1[index][:,:s])))
            untrans1c.append(seg[:,:s])
    data1 = np.concatenate(data1, axis=-1)
    untrans1c = np.concatenate(untrans1c)
    for index, seg in enumerate(tmp_untrans2):
        if seg.shape[-1] == tmp_trans2[index].shape[-1]:
            data2.append(np.vstack((seg, tmp_trans2[index])))
            untrans2c.append(seg)
        else:
            s = np.min([seg.shape[-1], tmp_trans2[index].shape[-1]])
            data2.append(np.vstack((seg[:,:s], tmp_trans2[index][:,:s])))
            untrans2c.append(seg[:,:s])
    data2 = np.concatenate(data2, axis=-1)
    untrans2c = np.concatenate(untrans2c)

    # data should be of shape (species*2, t*reps)
    print(np.shape(data1))
    dist = EnergyDistance(data1.T, data2.T, gpu=False)

    return dist