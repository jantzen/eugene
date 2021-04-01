# probability.py

import scipy
import numpy as np
from scipy.integrate import quad, dblquad
import warnings
import imp
from itertools import combinations
from functools import reduce

################################################################################
# Functions:
# Phi
# T
# SkewNorm
# SampleSkewNorm
# HellingerDistance
################################################################################

def Phi(x, m, s, a):
    return 0.5 * (1. + scipy.special.erf((x - m) / s / pow(2, 0.5)))


def T(h, a):
    f = lambda x: np.exp(-0.5 * pow(h, 2) * (1 + pow(x, 2))) / (1 + pow(x,2))
    temp = scipy.integrate.quad(f, 0, a)[0]
    return 1. / (2. * np.pi) * temp


def SkewNorm(x, m, s, a):
    return Phi(x, m, s, a) - 2 * T((x - m)/s, a)


def SampleSkewNorm(m, s, a):
    """ A quick and dirty implementation of a skew-normal random variable.
    Returns values from a skew-normal distribution with location m, scale s, and
    shape parameter a (see ). When a = 0, this is just a Gaussian with mean m
    and standard deviation, s.
    """
    # first, choose a random value in [0,1]:
    p = np.random.rand()

    # next, find the value of x corresponding that cumulative probability for
    # the skew-normal
    func = lambda x: p - SkewNorm(x, m, s, a)
    x = scipy.optimize.newton(func, 0)

    return x


def HellingerDistance(dist1, dist2, x_range=[-np.inf,np.inf]):    
    """ Computes the Hellinger distance between two univariate probability
    distributions, dist1 and dist2.
        inputs:
            dist1: a function that returns the probability of x
            dist2: a function that returns the probability of x
            x_range: a list of [low, high] values indicating the integration
            interval with respect to x (for computing the Hellinger distance)
	outputs:
            a scalar value representing the Hellinger distance.
    """

    func = lambda x: (np.sqrt(dist1(x) * dist2(x))).reshape(-1, 1)

    out = quad(func, x_range[0], x_range[1]) 

    h2 = 1. - out[0]

    hellinger = np.sqrt(h2)

    return hellinger


def Hellinger2D(dist1, dist2, x_low=-np.inf, x_high=np.inf, y_low=None,
        y_high=None):    
    """ Computes the Hellinger distance between two bivariate probability
    distributions, dist1 and dist2.
        inputs:
            dist1: a function that returns the probability of x, y
            dist2: a function that returns the probability of x, y
            x_low: float indicating the lower bound of the integration
            interval with respect to x (for computing the Hellinger distance)
	    x_high: float indicating the upper bound of the integration
            interval with respect to x (for computing the Hellinger distance)
	    y_low: a callable function describing the lower boundary curve of
            y as a function of x
	    y_high: a callable function describing the upper boundary curve of
            y as a function of x
        outputs:
            a scalar value representing the Hellinger distance.
    """

    if y_low == None:
        y_low = lambda x: x_low

    if y_high == None:
        y_high = lambda x: x_high

    if type(y_low) == np.float64 or type(y_low)==float:
        y_low = lambda x, y_low=y_low: y_low

    if type(y_high) == np.float64 or type(y_high)==float:
        y_high = lambda x, y_high=y_high: y_high
    
    func = lambda x,y: (np.sqrt(dist1(x,y) * dist2(x,y))).reshape(-1, 1)

    out = dblquad(func, x_low, x_high, y_low, y_high) 

    hellinger = 1. - np.sqrt(out[0])

    return hellinger


def EuclideanDistance(dist1, dist2, x_range=[-10,10]):
    """ Computes the Euclidean (L2) distance between two univariate probability
    distributions, dist1 and dist2.
        inputs:
            dist1: a function that returns the probability of x
            dist2: a function that returns the probability of x
            x_range: a list of [low, high] values indicating the integration
            interval with respect to x (for computing the Hellinger distance)
	outputs:
            a scalar value representing the L2 distance.
    """
    
    func = lambda x: (dist1(x) * dist2(x)).reshape(-1, 1)
    
    out = quad(func, x_range[0], x_range[1])
    
    l2 = np.sqrt(out)
    
    return l2


def EnergyDistance(X, Y, tol=10**(-12), gpu=False):
    """ Computes the energy distance (a statistical distance) between the
    cumulative distribution functions F and G of the independent random vectors
    X and Y.

    Inputs:
        X, Y: each is a s x d np-array, where d is the dimension of X and Y
        (assumed to be the same) and s is the number of samples
        tolerance: the threshold in relative error for deciding whether negative 
        values of D2 should count as 0, or result in an error

    Output:
        the energy distance, D, where:
        D^2(F,G) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||

    (see https://en.wikipedia.org/wiki/Energy_distance)
    """

    if len(X.shape) > 1 and X.shape[0] < X.shape[1]:
        warning_str = ("X appears to be transposed." + 
            " Shape of X: {}".format(X.shape))
        warnings.warn(warning_str)
    if len(Y.shape) > 1 and Y.shape[0] < Y.shape[1]:
        warning_str = ("Y appears to be transposed." + 
            " Shape of Y: {}".format(Y.shape))
        warnings.warn(warning_str)

    if len(X.shape) == 2 and len(Y.shape) == 2:
        n = X.shape[0]
        m = Y.shape[0]
    else:
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        n = X.shape[0]
        m = Y.shape[0]

    tmpX = X[np.all(np.isfinite(X), axis=1)]
    tmpY = Y[np.all(np.isfinite(Y), axis=1)]
    if not tmpX.shape == X.shape:
        errmsg = 'Warning: removing rows with bad values from X.'
        warnings.warn(errmsg)
    if not tmpY.shape == Y.shape:
        errmsg = 'Warning: removing rows with bad values from Y.'
        warnings.warn(errmsg)


    if gpu:
        try:
            imp.find_module('torch')
            found = True
        except ImportError:
            found = False
        if found:
            import torch
        else:
            gpu = False

    if gpu:
        if not torch.cuda.is_available():
            gpu = False
            errmsg = "No gpu available. Reverting to CPU method."
            warnings.warn(errmsg)

    if gpu:
        device = torch.device("cuda")

        n = torch.tensor(n, device=device, dtype=torch.double)
        m = torch.tensor(m, device=device, dtype=torch.double)
        Xg = torch.from_numpy(X).to(device)
        Yg = torch.from_numpy(Y).to(device)

        # Compute A = E||X - Y||
        A = torch.tensor(0., device=device, dtype=torch.double)
        for row in X:
            row = torch.from_numpy(row).to(device)
            diff = Yg - row
            norms = torch.sum(diff**2, -1)**(1./2.)
            A = A + torch.sum(norms,0)
        A /= (n * m)
        A = A.cpu().numpy()

        # Compute B = E||X - X'||
        B = torch.tensor(0., device=device, dtype=torch.double)
        for row in X:
            row = torch.from_numpy(row).to(device)
            diff = Xg - row
            norms = torch.sum(diff**2, -1)**(1./2.)
            B += torch.sum(norms,0)
        B /= (n * n)
        B = B.cpu().numpy()

        # Compute C = E||Y - Y'||
        C = torch.tensor(0., device=device, dtype=torch.double)
        for row in Y:
            row = torch.from_numpy(row).to(device)
            diff = Yg - row
            norms = torch.sum(diff**2, -1)**(1./2.)
            C += torch.sum(norms,0)
        C /= (m * m)
        C = C.cpu().numpy()

        # Compute energy distance
        D2 = 2. * A - B - C
        D = D2 ** (1./2.)

        if not np.isfinite(D):
            if abs(D2 / np.max([A, B + C]) / 2.) < tol:
                D = 0.
            else:
                raise ValueError("D^2 is negative ({0}) and " + 
                "this does not appear to be a machine precision issue.\n" +
                " A = {1}, B={2}, C={3}".format(D2,A,B,C))

        return D

    else:
        # Compute A = E||X - Y||
        A = 0.
        for row in X:
            diff = Y - row
            norms = np.sum(diff**2, axis=-1)**(1./2.)
            A += np.sum(norms)
        A /= (n * m)

        # Compute B = E||X - X'||
        B = 0.
        for row in X:
            diff = X - row
            norms = np.sum(diff**2, axis=-1)**(1./2.)
            B += np.sum(norms)
        B /= (n * n)

        # Compute C = E||Y - Y'||
        C = 0.
        for row in Y:
            diff = Y - row
            norms = np.sum(diff**2, axis=-1)**(1./2.)
            C += np.sum(norms)
        C /= (m * m)

        # Compute energy distance
        D2 = 2. * A - B - C
        D = D2 ** (1./2.)

        if not np.isfinite(D):
            if abs(D2 / np.max([A, B + C]) / 2.) < tol:
                D = 0.
            else:
                raise ValueError("D^2 is negative ({0}) and ".format(D2) +
                "this does not appear to be a machine precision issue.\n" +
                " A = {1}, B = {2}, C = {3}".format(D2,A,B,C))

        return D


def kSample(*args):
    """k-sample energy distance.

    Input:
        At least two samples (X, Y, ...)

    Output: 
        Computes energy distance for every pair of samples from args;
        returns the sum
    """
    pairs = combinations(args,2)
    return reduce(lambda a,b: a+b, [(EnergyDistance(tup[0],tup[1])) \
            for tup in pairs])


def nd_gaussian_pdf(mu, cov, points):
    """ Returns the value of the pdf of a multivariate Gaussian distribution
    with mean mu (of dimension vars x 1) and covariance cov (vars x vars) at the
    points in points (vars x num_points).
    """
    if ((type(mu) is float or type(mu) is np.float64) and (type(cov) is float
            or type(cov) is np.float64)):
        normalization = np.power(2. * np.pi * cov, -1./2.)
        diff = points - mu
        energies = -1./2. * np.power(diff, 2.) / cov
        out = normalization * np.exp(energies).reshape(1, -1)

    else:
        normalization = np.power(np.linalg.det(2. * np.pi * cov), -1./2.)
        diff = points - mu
        inv_cov = np.linalg.inv(cov)
        energies = -1./2. * np.sum(diff * np.dot(inv_cov, diff), axis=0)
        out = normalization * np.exp(energies).reshape(1,-1)

    return out


def significant(X, Y, D, n, B=1000.0, alpha = 0.05):
    """
    Implements the bootstrap hypothesis test described
    in Szekely and Rizzo 2004

    Inputs:
        X, Y: each is a s x d np-array, where d is the dimension of X and Y
        (assumed to be the same) and s is the number of samples
        D: energy distance obtained with EnergyDistance method
        n: how big each bootstrap sample should be
        B: number of bootstrap samples
        alpha: desired significance level

    Output:
        True/False: True if D was significant (reject the null hypothesis
        that X and Y are from the same distribution); False if not
    """
    # Make pooled sample
    W = np.concatenate((X, Y), axis=0)
    rows, cols = W.shape
    
    while not ((B+1) * alpha).is_integer():
        B+=1
    
    Sigma = 0
    for i in range(int(B)):
        sub = W[np.random.choice(rows, size=n*2, replace=False), :]
        W1 = sub[:n]
        W2 = sub[n:]
        D_B = EnergyDistance(W1, W2)
        if D > D_B: 
            Sigma+=1
            if Sigma/B > 1-alpha:
                return True

    return False
