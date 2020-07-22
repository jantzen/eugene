# file: test_metaparameters.py

from __future__ import division
import unittest
from eugene.src.data_prep.metaparameters import *
from eugene.src.dynamical_distance import *
import numpy as np
from scipy.integrate import ode
from scipy import zeros_like



class TestMetaparameters(unittest.TestCase):

    def setUp(self):
        # set up Kuramoto systems

        N = 3
        K1 = 0.

        omega= np.linspace(np.pi, 3. * np.pi / 2., N)

        def model(t, theta, arg1):
            K = arg1[0]
            omega = arg1[1]
            dtheta = zeros_like(theta)
            sum1 = [0.] * N
            for ii in range(N):
                for jj in range(N):
                    sum1[ii] += np.sin(theta[jj] - theta[ii])
                    
                dtheta[ii] = omega[ii] + (K / N) * sum1[ii]
            
            return dtheta

        # initial condition
        theta0 = np.linspace(0.1, 2., N)

        # time points
        t0 = 0.
        t1 = 100.
        resolution = 10000
        dt = (t1 - t0) / resolution

        # solve ODE1 at each timestep
        r = ode(model).set_integrator('lsoda')
        r.set_initial_value(theta0, t0).set_f_params([K1, omega])
        x = []
        t = []
        while r.successful() and r.t < t1:
            t.append(r.t)
            tmp = r.integrate(r.t+dt)
            x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))

        theta1 = np.concatenate(x, axis=1)

        # solve ODE2 at each timestep
        r = ode(model).set_integrator('lsoda')
        r.set_initial_value(theta0, t0).set_f_params([K1, omega * 2.])
        x = []
        t = []
        while r.successful() and r.t < t1:
            t.append(r.t)
            tmp = r.integrate(r.t+dt)
            x.append(np.array([np.cos(tmp), np.sin(tmp)]).reshape(-1,1))

        theta2 = np.concatenate(x, axis=1)

        c = 0.2

        theta1 += c * np.random.random_sample(theta1.shape)
        theta2 += c * np.random.random_sample(theta2.shape)

        self.timeseries = [theta1, theta2]


    def test_tune_ic_selection(self):
        print("Testing tune_passive_timeseries...")
        best_params, cost = tune_ic_selection(self.timeseries,
                parallel_compute=False)
        print("Best paramters: {}, cost: {}".format(best_params, cost))
        print("Types: {}, {}".format(type(best_params), type(cost)))
        print("Testing tune_passive_timeseries with parallel computation...")
        best_params_p, cost_p = tune_ic_selection(self.timeseries)
        print("Best paramters: {}, cost: {}".format(best_params, cost))
#        self.assertEqual(best_params, best_paramsp)
        self.assertEqual(cost, cost_p)


if __name__ == '__main__': 
    unittest.main() 
