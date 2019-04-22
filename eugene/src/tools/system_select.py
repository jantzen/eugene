# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:55:52 2019

@author: Colin
"""

import numpy as np
import sys
from os import makedirs, listdir
from os.path import isdir, join, exists
sys.path.insert(0, 'E:/User Data/GitHub/vabacon/')
sys.path.insert(0, 'E:/User Data/GitHub/AutomatedDiscovery/DADS2018/')
import approximate_kinds
import scipy.io
import pandas

def load_data(directory):
    # loads the data into a list
    set_names = listdir(directory)
    sets = []
    
    for set_name in set_names:
        set_path = join(directory, set_name)
        mat_data = scipy.io.loadmat(set_path)
        data_name = list(mat_data.keys())[-1]
        data = mat_data[data_name]
        sets.append(data.T)
    min_size = np.min([np.shape(x)[-1] for x in sets])
    for i, data in enumerate(sets):
        cut_data = np.take(data, range(min_size), -1)
        sets[i] = cut_data
    return sets


def test_data():
    data_list = []
    # make somewhere between 3 and 10 sets of data
    num_sets = np.random.randint(3, 10)
    for i in range(num_sets):
        # with a variable number of samples in each dataset
        num_samples = np.random.randint(100, 2000)
        data = np.random.random((3, num_samples))
        data_list.append(data)
    return data_list


def choose_data(sysA_dir, sysB_dir):
    # load dataA
#    dataA = test_data()
    dataA = load_data(sysA_dir)
    # load dataB
#    dataB = test_data()
    dataB = load_data(sysB_dir)
    # choose data from sysA using choose_trans_untrans
    #TODO: how to deal with data of different sample size?
    #   currently do this by cutting each dataset to smallest size
    untransA, transA = approximate_kinds.choose_untrans_trans([dataA], 1)
    # then choose data from sysB that is closest to untrans sysA
    # first, find the initial condition of the untransformed data
    # i.e. take the first entries on the last axis
    ic_untrans = np.take(untransA, [0], -1).flatten()
    # next, find the initial condition of the transformed data
    ic_trans = np.take(transA, [0], -1).flatten()
    
    untrans_dists = []
    trans_dists = []
    
    untransB_index = None
    transB_index = None
    for i, bset in enumerate(dataB):
        cur_ic = np.take(bset, [0], -1).flatten()
        untrans_dist = np.linalg.norm(ic_untrans-cur_ic)
        untrans_dists.append(untrans_dist)
        trans_dist = np.linalg.norm(ic_trans-cur_ic)
        trans_dists.append(trans_dist)
    
    untransB_index = np.argmin(untrans_dists)
    transB_index = np.argmin(trans_dists)
    # in the case that the two are the same, choose the second best for transB
    if untransB_index == transB_index:
        trans_dists[transB_index] = np.inf
        transB_index = np.argmin(trans_dists)
    
    untransB = dataB[untransB_index]
    transB = dataB[transB_index]
    
    if len(np.shape(untransA)) > 2:
        untransA = np.squeeze(untransA)
    if len(np.shape(transA)) > 2:
        transA = np.squeeze(transA)
    return {"untransA":untransA, "transA":transA, "untransB":untransB,
            "transB":transB}


if __name__ == '__main__':
    # systemsA systemsB output
    
    # Get systemsA folder
    sysA_dir = sys.argv[1]
    print(sysA_dir)
    assert isdir(sysA_dir), "Can't find Systems A directory."
    if not sysA_dir[-1] == '/':
        sysA_dir = sysA_dir + '/'
    sysA_names = listdir(sysA_dir)
    assert len(sysA_names) > 0, "Systems A directory is empty."
        
    # Get systemsB folder
    sysB_dir = sys.argv[2]
    assert isdir(sysB_dir), "Can't find Systems B directory."
    if not sysB_dir[-1] == '/':
        sysB_dir = sysB_dir + '/'
    sysB_names = listdir(sysB_dir)
    assert len(sysB_names) > 0, "Systems B directory is empty."

    # Get destination folder
    out_dir = sys.argv[3]
    if not exists(out_dir):
        makedirs(out_dir)

    selected = choose_data(sysA_dir, sysB_dir)
    # each selected is a dataset of dim x samples
    # put each dataset in a csv
    for name, data in selected.items():
        out_path = join(out_dir, name + ".csv")
        df = pandas.DataFrame(data)
        df.to_csv(out_path, index=False, header=False)
