import eugene as eu
import numpy as np
import warnings
import pdb

def test_choose_untrans_trans():
    # test 1-D
    t = np.linspace(0., 10., 100)
    x = []
    y = []
    z = []
    ics = np.random.normal(size=10000)
    for ic in ics:
        x.append(ic + np.exp(0.2 * t) - 1.)
    ics = np.random.normal(size=10000)
    for ic in ics:
        y.append(ic + np.exp(0.3 * t) - 1.)
    ics = np.random.normal(size=10000)
    for ic in ics:
        z.append(ic + np.exp(0.2 * t) + 0.2 * t -1.)

    data = [x, y, z]

    untrans, trans = eu.initial_conditions.choose_untrans_trans(data, 100)

    assert len(untrans[0]) == 100

    ## verify that warnings are properly triggered
    with warnings.catch_warnings(record=True) as w:
        x = []
        y = []
        ics = np.random.normal(size=10000)
        for ic in ics:
            x.append(ic + np.exp(0.2 * t) - 1.)
        ics = np.random.normal(size=10000) + 10.
        for ic in ics:
            y.append(ic + np.exp(0.3 * t) - 1.)

        data = [x, y]

        untrans, trans = eu.initial_conditions.choose_untrans_trans(data, 100)

    assert len(w) == 2

    # test 3-D
    t = np.concatenate([np.linspace(0., 10., 100).reshape(1,-1), np.linspace(0.,
        10., 100).reshape(1,-1),np.linspace(0., 10., 100).reshape(1,-1)],
        axis=0)
    x = []
    y = []
    z = []
    ics = np.random.normal(size=(3,10000))
    for ic in ics.T:
        ic = ic.reshape(-1,1)
        x.append(ic + np.exp(0.2 * t) - 1.)
    ics = np.random.normal(size=(3,10000))
    for ic in ics.T:
        ic = ic.reshape(-1,1)
        y.append(ic + np.exp(0.3 * t) - 1.)
    ics = np.random.normal(size=(3,10000))
    for ic in ics.T:
        ic = ic.reshape(-1,1)
        z.append(ic + np.exp(0.2 * t) + 0.2 * t -1.)

    data = [x, y, z]

    untrans, trans = eu.initial_conditions.choose_untrans_trans(data, 100)

    assert len(untrans[0]) == 100

    ## verify that warnings are properly triggered
    with warnings.catch_warnings(record=True) as w:
        x = []
        y = []
        ics = np.random.normal(size=(3,10000))
        for ic in ics.T:
            ic = ic.reshape(-1,1)
            x.append(ic + np.exp(0.2 * t) - 1.)
        ics = np.random.normal(size=(3,10000)) + 10.
        for ic in ics.T:
            ic = ic.reshape(-1,1)
            y.append(ic + np.exp(0.3 * t) - 1.)

        data = [x, y]

        untrans, trans = eu.initial_conditions.choose_untrans_trans(data, 100)

    assert len(w) == 2

