# damped_harmonic_oscillator.py
# https://anaconda.org/ijstokes/13-scientificpython/notebook

import numpy as np
from scipy.integrate import odeint
from math import pi
from eugene.src.tools.LVDSim import rangeCover


def dy(y, t, zeta, w0):
    """
    The right-hand side of the damped oscillator ODE
    """
    x, p = y[0], y[1]

    dx = p
    dp = -2 * zeta * w0 * p - w0 ** 2 * x

    return [dx, dp]


def get_data(times, zeta, y0=1.0, p0=0.0, w0=2.0):
    # initial state is y0, p0; initial position, and velocity.
    # time coordinate to solve the ODE for
    y0 = [y0, p0]
    # solve the ODE problem
    y1 = odeint(dy, y0, times, args=(zeta, w0*pi))
    return y1[:, 0]


def simData(params, max_time, num_times, overlay, stochastic_reps=None, range_cover=True):
    """ Generates data for a list of parameters corresponding to systems and
    returns a list of arrays of data that cover the same range. Initial velocity
    is always set to zero.

            Keyword arguments:
            params -- a list containing parameter sets. Each parameter set
                is a list that contains, in order: a damping coefficient,
                an original starting position, an original velocity, an original
                frequency (in fractions of pi), a transformed starting
                position, a transformed velocity, and a transformed frequency.
            max_time -- the highest time value to sample the system at.
            num_times -- the number of times to sample the system between t=0
                and t=max_time.
            overlay -- a function that takes an array of coordinates and returns
                a new data array. This function is overlaid on the data.

            Returns:
            2d array -- two arrays of data from the systems that cover the same
                range.
    """

    osci = []
    osci_trans = []
    if stochastic_reps is None:
        for param_set in params:
            osci.append([param_set[0], param_set[1], param_set[2], param_set[3]])
            osci_trans.append([param_set[0], param_set[4], param_set[5], param_set[6]])

    # TODO: Implement stochastic data generation

    times = []
    times_trans = []
    for i in range(len(osci)):
        times.append(np.linspace(0., max_time, num_times))
    for i in range(len(osci_trans)):
        times_trans.append(np.linspace(0., max_time, num_times))

    if stochastic_reps is None:
        ys = []
        ys_trans = []
        for i, sys in enumerate(osci):
            y = get_data(times[i], sys[0], y0=sys[1], p0=sys[2], w0=sys[3])
            ys.append(y)
        for i, sys in enumerate(osci_trans):
            y = get_data(times[i], sys[0], y0=sys[1], p0=sys[2], w0=sys[3])
            ys_trans.append(y)

        raw_data = []
        for i in range(len(osci)):
            f = overlay(ys[i])
            f_trans = overlay(ys_trans[i])
            raw_data.append([f, f_trans])

    # else:
    #     xys = []
    #     xys_trans = []
    #     for i, sys in enumerate(pend):
    #         temp = sys.check_xs(times[i])
    #         sys._x = sys._init_x
    #         for r in range(stochastic_reps):
    #             temp = np.vstack((temp, sys.check_xs(times[i])))
    #             sys._x = sys._init_x
    #         xys.append(temp)
    #
    #     for i, sys in enumerate(pend_trans):
    #         temp = sys.check_xs(times[i])
    #         sys._x = sys._init_x
    #         for r in range(stochastic_reps):
    #             temp = np.vstack((temp, sys.check_xs(times[i])))
    #             sys._x = sys._init_x
    #         xys_trans.append(temp)
    #
    #     raw_data = []
    #     for i in range(len(pend)):
    #         f = overlay(xys[i])
    #         f_trans = overlay(xys_trans[i])
    #         raw_data.append([f, f_trans])

    if range_cover:
        data, high, low = rangeCover(raw_data)


    return np.array(raw_data)