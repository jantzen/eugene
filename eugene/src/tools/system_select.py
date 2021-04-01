# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:55:52 2019

@author: Colin
"""

import numpy as np
import sys
from os import makedirs, listdir
from os.path import isdir, join, exists
sys.path.insert(0, '/home/colin/Documents/GitHub/autodisc/vabacon/')
sys.path.insert(0, '/home/colin/Documents/GitHub/autodisc/AutomatedDiscovery/DADS2018/')
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
        data_name = list(mat_data.keys())[0]
        data = np.asarray(mat_data[data_name])
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


def choose_pairs(sysA_dir, sysB_dir, sysC_dir):
    #TODO: make this generalized for any number of systems
    dataA = load_data(sysA_dir)
    dataB = load_data(sysB_dir)
    dataC = load_data(sysC_dir)
    
    ica1 = np.take(dataA[0], [0], -1).flatten()
    ica2 = np.take(dataA[1], [0], -1).flatten()
    icb1 = np.take(dataB[0], [0], -1).flatten()
    icb2 = np.take(dataB[1], [0], -1).flatten()
    icc1 = np.take(dataC[0], [0], -1).flatten()
    icc2 = np.take(dataC[1], [0], -1).flatten()
#    ica1 = dataA[0][0]
#    ica2 = dataA[1][0]
#    icb1 = dataB[0][0]
#    icb2 = dataB[1][0]
#    icc1 = dataC[0][0]
#    icc2 = dataC[1][0]
#    
    a1_b1_dist = np.linalg.norm(ica1-icb1)
    a1_b2_dist = np.linalg.norm(ica1-icb2)
    a1_b_min_index = np.argmin([a1_b1_dist, a1_b2_dist])
    
    a1_c1_dist = np.linalg.norm(ica1-icc1)
    a1_c2_dist = np.linalg.norm(ica1-icc2)
    a1_c_min_index = np.argmin([a1_c1_dist, a1_c2_dist])
    
    data_dict = {"untransA":dataA[0], "transA":dataA[1],
                 "untransB":dataB[a1_b_min_index],
                 "transB":dataB[not a1_b_min_index],
                 "untransC":dataC[a1_c_min_index],
                 "transC":dataC[not a1_c_min_index]}
    return data_dict


def choose_data(sysA_dir, sysB_dir):
    # load dataA
    dataA = test_data()
#    dataA = load_data(sysA_dir)
    # load dataB
    dataB = test_data()
#    dataB = load_data(sysB_dir)
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

    sysC_dir = None        
    if len(sys.argv) > 4:
        sysC_dir = sys.argv[3]
        assert isdir(sysC_dir), "Can't find Systems B directory."
        if not sysC_dir[-1] == '/':
            sysC_dir = sysC_dir + '/'
        sysC_names = listdir(sysC_dir)
        assert len(sysC_names) > 0, "Systems C directory is empty."
        out_dir = sys.argv[4]
        if not exists(out_dir):
            makedirs(out_dir)

    #selected = choose_data(sysA_dir, sysB_dir)
    selected = choose_pairs(sysA_dir, sysB_dir, sysC_dir)
    # each selected is a dataset of dim x samples
    # put each dataset in a csv
    for name, data in selected.items():
        out_path = join(out_dir, name + ".csv")
        df = pandas.DataFrame(data)
        df.to_csv(out_path, index=False, header=False)
