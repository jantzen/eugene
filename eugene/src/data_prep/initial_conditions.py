import copy
import numpy as np
import scipy.stats as stats
import warnings
from eugene.src.auxiliary.probability import EnergyDistance, nd_gaussian_pdf


def _choose_untrans_trans_1D(
        data_in, 
        num_reps, 
        alpha=0.5, 
        beta=0.2,
        mu_spec=None, # user-specified means for untrans, trans (2-element list)
        report=False):
    data = copy.deepcopy(data_in)

    num_conditions = len(data)

    pools = data

    if report:
        error_flag = np.zeros((num_conditions, num_conditions), dtype='int8') 

    # choose a pair of Gaussians from the joint data set (of initial values)

    # find the list of initials and their statistics
    ag_pool = []
    for pool in pools: 
        ag_pool = ag_pool + pool
    initials = []
    for segment in ag_pool:
        initials.append(segment[0])
    in_data = np.array(initials)
    mu = np.mean(in_data)
    var = np.var(in_data)

    # choose means and variance for distributions
    if mu_spec is not None:
        mu_untrans = mu_spec[0]
        mu_trans = mu_spec[1]
    else:
        errmsg = "Using auto-selected means for untrans and trans distributions."
        warnings.warn(errmsg)
        mu_untrans = mu - alpha * np.sqrt(var)
        mu_trans = mu + alpha * np.sqrt(var)
    var = beta * var 

    untrans = []
    trans = []

    untrans_scores = []
    trans_scores = []

    for pool in pools:
        densities = []
        for segment in pool:
            densities.append(nd_gaussian_pdf(mu_untrans, var,
                segment[0])[0][0])
        densities = np.array(densities)
        untrans_scores.append(np.sort(densities)[-num_reps:])
        indices = np.argsort(densities)[-num_reps:]
        tmp = []
        for index in indices:
            tmp.append(pool[index])
        for index in sorted(indices, reverse=True):
            del pool[index]
        untrans.append(tmp)

    for pool in pools:
        densities = []
        for segment in pool:
            densities.append(nd_gaussian_pdf(mu_trans, var,
                segment[0])[0][0])
        densities = np.array(densities)
        trans_scores.append(np.sort(densities)[-num_reps:])
        indices = np.argsort(densities)[-num_reps:]
        tmp = []
        for index in indices:
            tmp.append(pool[index])
        trans.append(tmp)

    # REPLACE THIS WITH HIGH-DIMENSIONAL TWO-SAMPLE TEST FOR DIFFERENCE OF MEANS

    # test quality of sampled distributions
    # test untrans
    nn = len(untrans_scores)
    for ii in range(nn):
        for jj in range(ii, nn):
            result = stats.ks_2samp(untrans_scores[ii], untrans_scores[jj])
            if result.pvalue < 0.005:
                errmsg = """Warning: the untrans initial conditions for
                conditions {} and {} do not match (p < 0.005).""".format(ii, jj)
                warnings.warn(errmsg)
                if report:
                    error_flag[ii,jj] += 1
                    error_flag[jj,ii] += 1

    # test trans
    nn = len(trans_scores)
    for ii in range(nn):
        for jj in range(ii, nn):
            result = stats.ks_2samp(trans_scores[ii], trans_scores[jj])
            if result.pvalue < 0.005:
                errmsg = """Warning: the trans initial conditions for
                conditions {} and {} do not match (p < 0.005).""".format(ii, jj)
                warnings.warn(errmsg)
                if report:
                    error_flag[ii,jj] += 2
                    error_flag[jj,ii] += 2

    if report:
        return untrans, trans, error_flag
    else:
        return untrans, trans


def choose_untrans_trans(data_in, num_reps, alpha=0.5, beta=0.2, mu_spec=None, report=False):
    """ Input:
        data: A list of lists of numpy ndarrays, one for each condition. They are expected to 
        be dim x p where dim is the dimension and p the number of samples.
        num_reps: The desired (approximate) number of segments to end up
            in each group
        alpha:
        beta:
        mu: a list of arrays, each of dim x 1 dimensions
        Output:
        untrans: a list of lists (one for each condition)
        trans: a list of lists (one for each condition)
    """
    data = copy.deepcopy(data_in)

    num_conditions = len(data)

    if data[0][0].ndim == 1 or min(data[0][0].shape)==1:
        print("Choosing untrans and trans segments for 1-D data...")
        return _choose_untrans_trans_1D(data_in, num_reps, alpha=alpha,
                beta=beta, mu_spec=mu_spec, report=report)

    # check the segments for proper format
    for j, subset in enumerate(data):
        dim = np.min(subset[0].shape)

        for i, seg in enumerate(subset):
            shape = seg.shape
            a = np.min(shape)
            b = np.max(shape)
            if not a == dim:
                raise ValueError("All segments must have the same dimension.")
            if not a == shape[0]:
                data[j][i] = seg.T

    pools = data

    if report:
        error_flag = np.zeros((num_conditions, num_conditions), dtype='int8') 

    # choose a pair of Gaussians from the joint data set (of initial values)

    # find the list of initials and their statistics
    ag_pool = []
    for pool in pools: 
        ag_pool = ag_pool + pool
    initials = []
    for segment in ag_pool:
        initials.append(segment[:,0].reshape(-1,1))
    initials = tuple(initials)
    in_data = np.concatenate(initials, axis=1)
    mu = np.mean(in_data, axis=1).reshape(-1,1)
    cov = np.cov(in_data)

    # test quality of covariance matrix
    det = np.linalg.det(2. * np.pi * cov)
    if det <= 0.0:
        errmsg = "Rank deficient covariance matrix. Number of fragments: {}".format(
            len(initials)) + "  Mu = {}, Cov = {}, |Cov| = {}".format(mu, cov, det)
        raise Warning(errmsg)

    # find the largest eigenvector-eigenvalue pair
    w, v = np.linalg.eig(cov)
    w_max = np.sqrt(np.max(w))
    e_vec = v[:,np.argsort(w)[-1]].reshape(-1,1)

    # choose means and covariance matrix for distributions
    if mu_spec is not None:
        mu_untrans = mu_spec[0]
        mu_trans = mu_spec[1]
    else:
        errmsg = "Using auto-selected means for untrans and trans distributions."
        warnings.warn(errmsg)
        mu_untrans = mu - alpha * w_max * e_vec
        mu_trans = mu + alpha * w_max * e_vec
    u, s, v = np.linalg.svd(cov)
    s = beta * s
    cov_t = np.dot(np.dot(u, np.diag(s)), v)

    peak_untrans = nd_gaussian_pdf(mu_untrans, cov_t, mu_untrans)
    peak_trans = nd_gaussian_pdf(mu_trans, cov_t, mu_trans)

    untrans = []
    trans = []

    untrans_scores = []
    trans_scores = []

    for pool in pools:
        densities = []
        for segment in pool:
            densities.append(nd_gaussian_pdf(mu_untrans, cov_t,
                segment[:,0].reshape(-1,1))[0][0])
        densities = np.array(densities)
        untrans_scores.append(np.sort(densities)[-num_reps:])
        indices = np.argsort(densities)[-num_reps:]
        tmp = []
        for index in indices:
            tmp.append(pool[index])
        for index in sorted(indices, reverse=True):
            del pool[index]
        untrans.append(tmp)

    for pool in pools:
        densities = []
        for segment in pool:
            densities.append(nd_gaussian_pdf(mu_trans, cov_t,
                segment[:,0].reshape(-1,1))[0][0])
        densities = np.array(densities)
        trans_scores.append(np.sort(densities)[-num_reps:])
        indices = np.argsort(densities)[-num_reps:]
        tmp = []
        for index in indices:
            tmp.append(pool[index])
        trans.append(tmp)

    # test the quality of the overlap between treatments in both untrans and
    # trans pools (in other words, how indistinguishable are the initial
    # distributions)

    # REPLACE THIS WITH HIGH-DIMENSIONAL TWO-SAMPLE TEST FOR DIFFERENCE OF MEANS

    # test quality of sampled distributions
    # test untrans
    nn = len(untrans_scores)
    for ii in range(nn):
        for jj in range(ii, nn):
            result = stats.ks_2samp(untrans_scores[ii], untrans_scores[jj])
            if result.pvalue < 0.005:
                errmsg = """Warning: the untrans initial conditions for
                conditions {} and {} do not match (p < 0.005).""".format(ii, jj)
                warnings.warn(errmsg)
                if report:
                    error_flag[ii,jj] += 1
                    error_flag[jj,ii] += 1

    # test trans
    nn = len(trans_scores)
    for ii in range(nn):
        for jj in range(ii, nn):
            result = stats.ks_2samp(trans_scores[ii], trans_scores[jj])
            if result.pvalue < 0.005:
                errmsg = """Warning: the trans initial conditions for
                conditions {} and {} do not match (p < 0.005).""".format(ii, jj)
                warnings.warn(errmsg)
                if report:
                    error_flag[ii,jj] += 2
                    error_flag[jj,ii] += 2

    if report:
        return untrans, trans, error_flag
    else:
        return untrans, trans


