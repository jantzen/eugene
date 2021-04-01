# -*- coding: utf-8 -*-
# multiPendulum.py
# https://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/

import numpy as np

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify
from scipy.integrate import odeint
from eugene.src.tools.LVDSim import rangeCover

def integrate_pendulum(n, times,
                       initial_positions=135,
                       initial_velocities=0,
                       lengths=None, masses=1):
    """Integrate a multi-pendulum with `n` sections"""
    #-------------------------------------------------
    # Step 1: construct the pendulum model
    
    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass) 
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')
    
    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x))
        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(forces, particles)
    
    #-----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

    # initial positions and velocities – assumed to be given in degrees
    y0 = np.deg2rad(np.concatenate([np.broadcast_to(initial_positions, n),
                                    np.broadcast_to(initial_velocities, n)]))
        
    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation 
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(y, t, args):
        vals = np.concatenate((y, args))
        sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
        return np.array(sol).T[0]

    # ODE integration
    return odeint(gradient, y0, times, args=(parameter_vals,))
    

def get_xy_coords(p, lengths=None):
    """Get (x, y) coordinates from generalized coordinates p"""
    p = np.atleast_2d(p)
    n = p.shape[1] // 2
    if lengths is None:
        lengths = np.ones(n) / n
    zeros = np.zeros(p.shape[0])[:, None]
    x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
    return np.cumsum(x, 1), np.cumsum(y, 1)
    

def get_data(pendulums, t_0=0, t_elapsed=10, delta_t=1000):
    """Returns (x, y) coords for each arm of the defined pendulum"""
    t = np.linspace(t_0, t_elapsed, delta_t)
    p = integrate_pendulum(n=pendulums, times=t)
    return get_xy_coords(p)


def simData(params, max_time, num_times, overlay, stochastic_reps=None, range_cover=True):
    """ Generates data for a list of parameters corresponding to systems and
    returns a list of arrays of data that cover the same range. Initial velocity
    is always set to zero.

            Keyword arguments:
            params -- a list containing parameter sets. Each parameter set
                is a list that contains, in order: a number of pendulums,
                an original starting position, a transformed starting position,
                and an array of arm lengths.
            max_time -- the highest time value to sample the system at.
            num_times -- the number of times to sample the system between t=0
                and t=max_time.
            overlay -- a function that takes an array of coordinates and returns
                a new data array. This function is overlaid on the data.

            Returns:
            2d array -- two arrays of data from the systems that cover the same
                range.
    """

    pend = []
    pend_trans = []
    if stochastic_reps is None:
        for param_set in params:
            pend.append([param_set[0], param_set[1], param_set[3]])
            pend_trans.append([param_set[0], param_set[2], param_set[3]])

    # TODO: Implement stochastic data generation

    times = []
    times_trans = []
    for i in range(len(pend)):
        times.append(np.linspace(0., max_time, num_times))
    for i in range(len(pend_trans)):
        times_trans.append(np.linspace(0., max_time, num_times))

    if stochastic_reps is None:
        xys = []
        xys_trans = []
        for i, sys in enumerate(pend):
            p = integrate_pendulum(n=sys[0], times=times[i],
                                   initial_positions=sys[1], lengths=sys[2])
            xy = np.array(get_xy_coords(p, lengths=np.array(sys[2])))
            xs = xy[0,:,2]
            ys = xy[1,:,2]
            xy = [xs, ys]
            xys.append(xy)
        for i, sys in enumerate(pend_trans):
            p = integrate_pendulum(n=sys[0], times=times[i],
                                   initial_positions=sys[1], lengths=sys[2])
            xy = np.array(get_xy_coords(p, lengths=np.array(sys[2])))
            xs = xy[0, :, 2]
            ys = xy[1, :, 2]
            xy = [xs, ys]
            xys_trans.append(xy)

        raw_data = []
        for i in range(len(pend)):
            f = overlay(xys[i])
            f_trans = overlay(xys_trans[i])
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