# file: test_dd_scan.py

from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from eugene.src.dd_scan import *
from scipy.io import matlab
from scipy import zeros_like
import pickle


def f(t, theta, args):
    """ Kuramoto system.
        args[0]: K
        args[1]: list of frequencies, omega
    """
    dtheta =zeros_like(theta)
    summand = [0] * 3
    for ii in range(3):
        for jj in range(3):
            summand[ii] += np.sin(theta[jj] - theta[ii])
        dtheta[ii] = args[1][ii] + (args[0] / 3) * summand[ii]
    return dtheta


def test_DiffScanner():
    # parameters
    K = 1.0

    omega= np.linspace(np.pi, 3. * np.pi / 2., 3)
    add_noise = True
    std = 0.05
    alpha = 1.0

    # initial condition
    t0 = 0.
    theta0 =  np.linspace(0.1, 2., 3)

    dt = 5. * 10**(-4)

    t1 = 1000. # onset of first transition
    t2 = 2000. # onset of second transition
    tf = 3000.0 # final time

    # properties of the transition
    dK = -1.0 
    domega = omega 

    r1 = ode(f).set_integrator('lsoda')
    r1.set_initial_value(theta0, t0).set_f_params([K, omega])

    times = []
    theta_vals = []

    times.append(t0)
    theta_vals.append(theta0.reshape(1,-1))

    while r1.successful() and r1.t < t1:
        times.append(r1.t+dt)
        theta_vals.append(r1.integrate(r1.t+dt).reshape(1,-1))

    K += dK
    r2 = ode(f).set_integrator('lsoda')
    r2.set_initial_value(r1._y, r1.t).set_f_params([K, omega])

    while r2.successful() and t1 <= r2.t < t2:
        times.append(r2.t+dt)
        theta_vals.append(r2.integrate(r2.t+dt).reshape(1,-1))

    omega += domega
    r3 = ode(f).set_integrator('lsoda')
    r3.set_initial_value(r2._y, r2.t).set_f_params([K, omega])

    while r3.successful() and t2 <= r3.t < tf:
        times.append(r3.t+dt)
        theta_vals.append(r3.integrate(r3.t+dt).reshape(1,-1))

    theta_vals = np.concatenate(theta_vals, axis=0)

    x_vals = np.cos(theta_vals)
    y_vals = np.sin(theta_vals)

    # add noise
    if add_noise:
        x_vals = x_vals + np.random.normal(0., std, size = x_vals.shape)
        y_vals = y_vals + np.random.normal(0., std, size = y_vals.shape)

    xy_vals = np.concatenate([x_vals, y_vals], axis=1)

    width_times = np.asarray([250.])
    lag_times = np.asarray([50.])
    step_time = 23.

    step_size = int(round(step_time / dt)) 
    widths = np.round(width_times / dt).astype('int')
    lags = np.round(lag_times / dt).astype('int')

    for window_width in widths:
        for lag in lags:
            data = xy_vals.T
            diff = DiffScanner(data, window_width=window_width, step_size=step_size, lag=lag)
            print('starting diff scan...')
            frags = window_width / 200 
            reps = frags / 10
            diff.start_scan(frags=frags, reps=reps, free_cores=4, alpha=alpha)
            diff_times = []
            for ii in diff._scan[:,0]:
                index = int(ii + window_width + lag / 2 - 1)
                diff_times.append(times[index])

        max_index = np.argmax(diff._scan[:,1])
        time_max = diff_times[max_index]
        assert 900. < time_max and time_max < 1100.

    for window_width in widths:
        for lag in lags:
            data = xy_vals[:,:3].T
            diff = DiffScanner(data, window_width=window_width, step_size=step_size, lag=lag)
            print('starting diff scan...')
            frags = window_width / 200 
            reps = frags / 10
            diff.start_scan(frags=frags, reps=reps, free_cores=4, alpha=alpha)
            diff_times = []
            for ii in diff._scan[:,0]:
                index = int(ii + window_width + lag / 2 - 1)
                diff_times.append(times[index])

        max_index = np.argmax(diff._scan[:,1])
        time_max = diff_times[max_index]
        assert 900. < time_max and time_max < 1100.

