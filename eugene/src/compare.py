import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
#import warnings
import math
import copy
import pdb

#Classes:
# SymModel

#Functions:
#  FitPolyCV
#  BuildSymModel
#  CompareModels


###################################################################
###################################################################
#Classes


class SymModel( object ):
    """ An object holding a collection of polynomial models, each 
        corresponding to a symmetry labeled by a distinct value of what
        is presumed to be a single, real-valued parameter - in other 
        words, 
        An object holding a collection of members of a Lie group of symmetries. 
        Each polynomial is represented by an ndarray containing the 
        polynomial coefficients, highest power first.
        sampled_data: an r x 2 array of values to which the corresponding
        polynomials were fit
    """
    
    def __init__(self, index_var, target_var, sys_id, sampled_data, polynomials = [], epsilon=None):
        self._index_var = index_var
        self._target_var = target_var
        self._sys_id = sys_id
        self._sampled_data = sampled_data
        self._polynomials = polynomials
        self._epsilon = epsilon

    def Update(self, polynomials):
        self._polynomials = polynomials


class nPolynomial( object ):
    """ A representation of a polynmial of degree d and number of variables n.
    """
    
    def __init__(self, n, d, params=None):
        self._num_vars = n
        self._degree = d
        if params == None:
            self._params = np.zeros(math.factorial(n +
                d) / math.factorial(d) / math.factorial(n))
        else:
            self._params = params
       

        self._exponents = exponents
         

#    def Eval(self, x, params=None):
#        # Evaluate the polynomial given params at the point x
#        if params == None:
#            params = self._params

           


        


###################################################################
###################################################################
#Functions


def exponents(n, d):
    """ This function computes the exponents (as n-tuples) for each term in a
        polynomial in n variables of degree d.
    """
    try:
        assert type(n) == int and n > 0 and type(d) == int and d>0
    
        if n == 1:
            out = []
            for i in range(d+1):
                out.append([i])
            return out
        else:
            out_old = exponents(n-1, d)
            out_new = []
            for i in range(len(out_old)):
                for j in range(d+1):
                    if sum(out_old[i]) + j <= d:
                        temp = copy.deepcopy(out_old[i])
                        temp.extend([j])
                        out_new.append(temp)
                    else:
                        break
            return out_new
    except AssertionError:
        print "Improper input to exponents function. n and d must be positive integers."


def npoly_val(params, exponents, x_vals):
    """ Returns the value of an n-variable polynomial at the point given by
        x_vals
    """
    sum_of_terms = 0
    for i, param in enumerate(params):
        term = param
        for j, x in enumerate(x_vals): 
            term = term * pow(x, exponents[i][j])
        sum_of_terms += term

    return sum_of_terms


def residuals(params, exponents, xdata, ydata):
    """ For use with surface_fit. Given a polynomial in n variables, a set of n-dimensional xdata, 
        and a set of ydata, compute the residuals with respect to the y values
        computed from the poynomial.
    """
    resids = []
    num_vars = len(xdata)
    for i, y in enumerate(ydata):
        point_val = []
        for j in range(num_vars):
            point_val.append(xdata[j][i])
        expected = npoly_val(params, exponents, point_val)
        resids.append(y - expected)
     
    return resids


def surface_fit(xdata, ydata, order):
    """ Takes a set of x0, x1, ..., xn, y data in the form of n+1 np arrays  (with the
        indpendent variable in the first n) and fits a polynomial surface of the
        indicated order using least-squares.
    """
    # compute necessary number of params. For a polynomial in n variables of
    # degree d, there are (n+d)C(d) terms
    num_vars = len(xdata) 
    num_params = math.factorial(num_vars +
            order)/math.factorial(num_vars)/math.factorial(order)
    params_guess = np.ones(num_params)

    # get the exponents for each term
    exps = exponents(num_vars, order)

    params = opt.leastsq(residuals, params_guess, args=(exps, xdata, ydata))

    return params
        

def FitPolyCV(data, epsilon=0):
    """ Takes a set of x,y data in the form of n+1 columns (x in the first n, y in the
        last) and fits a polynomial surface in n-variables of order determined by 10-fold
        cross-validation.
    """

    # Determine the number of independent variables
    x_vars = data.shape[1] - 1
 
    # Turn off warnings from polyfit
#    warnings.simplefilter('ignore', np.RankWarning)

    np.random.shuffle(data)
    partition = np.array_split(data, 10)

    # first try a linear fit
    order = 1
        
    # compute the error using each partition as validation set
    fit_residuals = []
    for p in range(len(partition)):
        # build training data
        training_set = np.empty([0,(x_vars + 1)],dtype='float64')
        for i in range(len(partition)):
            if i != p:
                training_set = np.concatenate((training_set, partition[i]), 0)
        
        # fit polynomial surface in x_vars number of variables
        
        # restructure the xdata into a list of np.arrays
        x = []
        for i in range(x_vars):
            x.append(training_set[:,i])
        y = training_set[:, x_vars]
        # Note: the n-variable polynomial surface is represented by a list of
        # parameters. The terms to which each parameter corresponds is
        # implicitly given by the order in which the exponents function produces
        # terms for d-order polynomial in n variables.
        params, cov = surface_fit(x, y, order)

        # compute error of fit against partitition p
        x = []
        for i in range(x_vars):
            x.append(partition[p][:,i])
        y = partition[p][:,x_vars]
        fit_residuals.extend(residuals(params, exponents(x_vars, order), x, y))

    # compute mean square error for first order
    mse = np.mean(pow(np.array(fit_residuals),2))
    best_fit_order = order

    # assess fits for higher-order polynomials until the minimum is passed
    loop = True

#    pdb.set_trace()

    while loop:
        order += 1
        # partition the data
        np.random.shuffle(data)
        partition = np.array_split(data, 10)

        # compute the error using each partition as validation set
        fit_residuals = []
        for p in range(len(partition)):
            # build training data
            training_set = np.empty([0,(x_vars + 1)],dtype='float64')
            for i in range(len(partition)):
                if i != p:
                    training_set = np.concatenate((training_set, partition[i]), 0)

            # fit polynomial surface in x_vars number of variables
            # restructure the xdata into a list of np.arrays
            x = []
            for i in range(x_vars):
                x.append(training_set[:,i]) 
            y = training_set[:,x_vars]
            params, cov = surface_fit(x, y, order)

            # compute error of fit against partition p
            x = []
            for i in range(x_vars):
                x.append(partition[p][:,i])
            y = partition[p][:,x_vars]
            fit_residuals.extend(residuals(params, exponents(x_vars, order), x, y))
   
        # compute mean square error for current order
        mse_candidate = np.mean(pow(np.array(fit_residuals),2))

        # if significantly better, keep it. If not, keep the old and halt.
        if (mse - mse_candidate) / mse > epsilon:
            mse = mse_candidate
            best_fit_order = order

        else:
            loop = False
            
        # cap the complexity
        if order >= 10:
            loop = False

    # using the best-fit order, fit the full data set
    x = []
    for i in range(x_vars):
        x.append(data[:,i])
    y = data[:,x_vars]
    best_fit_params, cov = surface_fit(x, y, best_fit_order)

    return best_fit_params

 
def BuildSymModel(data_frame, index_var, target_var, sys_id, epsilon=0):
    # from the raw curves of target vs. index, build tranformation 
    # curves (e.g., target1 vs. target2)
    abscissa = data_frame._target_values[0]
    ordinates = data_frame._target_values[1:]
    
    # for each curve, fit a polynomial using 10-fold cross-validation
    # to choose the order
    polynomials = []
    sampled_data = []

    for curve in ordinates:
        # format the data
        data = np.vstack((abscissa, curve))
        data = data.transpose()

        # fit a polynomial of the best order determined by 10-fold CV
        best_fit = FitPolyCV(data, epsilon)

        # add to data arrays to pass out in final SymModel
        polynomials.append(best_fit)
        sampled_data.append(data)

    # build and output a SymModel object
    return SymModel(index_var, target_var, sys_id, sampled_data, polynomials, 
            epsilon)


def CompareModels(model1, model2, alpha=1.):
    """ Tests whether the models (and the systems they model) are equivalent.
    Returns 0 if they are indistinguishable (of the same dynamical kind as far
    as we can tell) and 1 otherwise.
    """

    # Turn off warnings from polyfit
    warnings.simplefilter('ignore', np.RankWarning)

    # initialize containers for data that may be passed out
    combined_sampled_data = []
    combined_polynomials = []

    # PREPARE THE DATA
    data1 = []
    data2 = []

    # for each sampled curve, randomize the data
    for curve in model1._sampled_data:
        np.random.shuffle(curve)
        data1.append(curve)
    data1 = np.concatenate(data1, 1)

    for curve in model2._sampled_data:
        np.random.shuffle(curve)
        data2.append(curve)
    data2 = np.concatenate(data2, 1)

    # in order to estimate the base error when both systems belong to the same
    # kind, separate each data set in two (and treat each pair as its own
    # comparison problem)
    base_error_data1 = np.array_split(data1, 2)
    base_error_data2 = np.array_split(data2, 2)

    epsilon = min(model1._epsilon, model2._epsilon)

    # OUTER CROSS-VALIDATION LOOP
    ############################################################################
    
    # Partition the data
    partition1 = np.array_split(data1, 10)
    partition2 = np.array_split(data2, 10)

    base_error_partition1a = np.array_split(base_error_data1[0], 10)
    base_error_partition1b = np.array_split(base_error_data1[1], 10)
    base_error_partition2a = np.array_split(base_error_data2[0], 10)
    base_error_partition2b = np.array_split(base_error_data2[1], 10)

    # Prepare empty arrays to store squared errors for each test partition
    SE_sep = []
    SE_joint = []

    SE_base_error1_sep = []
    SE_base_error1_joint = []
    SE_base_error2_sep = []
    SE_base_error2_joint = []

    # Loop over partitions, using each as a test set once
    for p in range(10):
        # Build training data for each predictive model (separate vs. joint)
        cols = np.shape(partition1[0])[1]
        training_set_sep1 = np.empty([0, cols],dtype='float64')
        training_set_sep2 = np.empty([0, cols],dtype='float64')
        training_set_joint = np.empty([0, cols],dtype='float64')

        training_set_base_error1a = np.empty([0, cols], dtype='float64')
        training_set_base_error1b = np.empty([0, cols], dtype='float64')
        training_set_base_error1_joint = np.empty([0, cols], dtype='float64')
        training_set_base_error2a = np.empty([0, cols], dtype='float64')
        training_set_base_error2b = np.empty([0, cols], dtype='float64')
        training_set_base_error2_joint = np.empty([0, cols], dtype='float64')
 
        for i in range(10):
            if i != p:
                training_set_sep1 = np.concatenate((training_set_sep1,
                    partition1[i]), 0)
                training_set_sep2 = np.concatenate((training_set_sep2,
                    partition2[i]), 0)
                training_set_joint = np.concatenate((training_set_joint,
                    partition1[i], partition2[i]), 0)

                training_set_base_error1a = np.concatenate((training_set_base_error1a,
                    base_error_partition1a[i]),0)
                training_set_base_error1b = np.concatenate((training_set_base_error1b,
                    base_error_partition1b[i]),0)
                training_set_base_error1_joint = np.concatenate((
                    training_set_base_error1_joint, base_error_partition1a[i], 
                    base_error_partition1b[i]), 0)
                training_set_base_error2a = np.concatenate((training_set_base_error2a,
                    base_error_partition2a[i]),0)
                training_set_base_error2b = np.concatenate((training_set_base_error2b,
                    base_error_partition2b[i]),0)
                training_set_base_error2_joint = np.concatenate((
                    training_set_base_error2_joint, base_error_partition2a[i], 
                    base_error_partition2b[i]), 0)
 

        # TRAIN THE PREDICTORS
        ########################################################################
        polynomials_sep1 = []
        polynomials_sep2 = []
        polynomials_joint = []

        polynomials_base_error1a = []
        polynomials_base_error1b = []
        polynomials_base_error1_joint = []
        polynomials_base_error2a = []
        polynomials_base_error2b = []
        polynomials_base_error2_joint = []
        
        # Loop over curves in the training set
        for index in range(cols/2):
            x_col = 2 * index
            curve_sep1 = training_set_sep1[:,x_col:(x_col+2)]
            curve_sep2 = training_set_sep2[:,x_col:(x_col+2)]
            curve_joint = training_set_joint[:,x_col:(x_col+2)]

            polynomials_sep1.append(FitPolyCV(curve_sep1, epsilon))
            polynomials_sep2.append(FitPolyCV(curve_sep2, epsilon))
            polynomials_joint.append(FitPolyCV(curve_joint, epsilon))

            curve_base_error1a = training_set_base_error1a[:,x_col:(x_col+2)]
            curve_base_error1b = training_set_base_error1b[:,x_col:(x_col+2)]
            curve_base_error1_joint = training_set_base_error1_joint[:,x_col:(x_col+2)]
            curve_base_error2a = training_set_base_error2a[:,x_col:(x_col+2)]
            curve_base_error2b = training_set_base_error2b[:,x_col:(x_col+2)]
            curve_base_error2_joint = training_set_base_error2_joint[:,x_col:(x_col+2)]
 
            polynomials_base_error1a.append(FitPolyCV(curve_base_error1a,
                epsilon))
            polynomials_base_error1b.append(FitPolyCV(curve_base_error1b,
                epsilon))
            polynomials_base_error1_joint.append(FitPolyCV(curve_base_error1_joint, 
                epsilon))
            polynomials_base_error2a.append(FitPolyCV(curve_base_error2a,
                epsilon))
            polynomials_base_error2b.append(FitPolyCV(curve_base_error2b,
                epsilon))
            polynomials_base_error2_joint.append(FitPolyCV(curve_base_error2_joint, 
                epsilon))
 
            
        # TEST THE PREDICTORS
        ########################################################################
        # Loop over curves in the test set
        for index in range(cols/2):
            x_col = 2 * index
            x_sep1 = partition1[p][:,x_col]
            y_sep1 = partition1[p][:,(x_col+1)]
            x_sep2 = partition2[p][:,x_col]
            y_sep2 = partition2[p][:,(x_col+1)]
            x_joint = np.concatenate((partition1[p][:,x_col], 
                partition2[p][:,x_col]))
            y_joint = np.concatenate((partition1[p][:,(x_col+1)], 
                partition2[p][:,(x_col+1)]))

            x_base_error1a = base_error_partition1a[p][:,x_col]
            y_base_error1a = base_error_partition1a[p][:,(x_col+1)]
            x_base_error1b = base_error_partition1b[p][:,x_col]
            y_base_error1b = base_error_partition1b[p][:,(x_col+1)]
            x_base_error1_joint = np.concatenate((
                base_error_partition1a[p][:,x_col],
                base_error_partition1b[p][:,x_col]))
            y_base_error1_joint = np.concatenate((
                base_error_partition1a[p][:,(x_col+1)],
                base_error_partition1b[p][:,(x_col+1)]))
            x_base_error2a = base_error_partition2a[p][:,x_col]
            y_base_error2a = base_error_partition2a[p][:,(x_col+1)]
            x_base_error2b = base_error_partition2b[p][:,x_col]
            y_base_error2b = base_error_partition2b[p][:,(x_col+1)]
            x_base_error2_joint = np.concatenate((
                base_error_partition2a[p][:,x_col],
                base_error_partition2b[p][:,x_col]))
            y_base_error2_joint = np.concatenate((
                base_error_partition2a[p][:,(x_col+1)],
                base_error_partition2b[p][:,(x_col+1)]))
            

            for i in range(len(x_sep1)):
                SE_sep.append((np.polyval(polynomials_sep1[index],
                        x_sep1[i]) - y_sep1[i])**2)

            for i in range(len(x_sep2)):
                SE_sep.append((np.polyval(polynomials_sep2[index],
                        x_sep2[i]) - y_sep2[i])**2)
            
            for i in range(len(x_joint)):
                SE_joint.append((np.polyval(polynomials_joint[index],
                    x_joint[i]) - y_joint[i])**2)


            for i in range(len(x_base_error1a)):
                SE_base_error1_sep.append((np.polyval(polynomials_base_error1a[index],
                    x_base_error1a[i]) - y_base_error1a[i])**2)

            for i in range(len(x_base_error1b)):
                SE_base_error1_sep.append((np.polyval(polynomials_base_error1b[index],
                    x_base_error1b[i]) - y_base_error1b[i])**2)

            for i in range(len(x_base_error1_joint)):
                SE_base_error1_joint.append((np.polyval(polynomials_base_error1_joint[index],
                    x_base_error1_joint[i]) - y_base_error1_joint[i])**2)

            for i in range(len(x_base_error2a)):
                SE_base_error2_sep.append((np.polyval(polynomials_base_error2a[index],
                    x_base_error2a[i]) - y_base_error2a[i])**2)

            for i in range(len(x_base_error2b)):
                SE_base_error2_sep.append((np.polyval(polynomials_base_error2b[index],
                    x_base_error2b[i]) - y_base_error2b[i])**2)

            for i in range(len(x_base_error2_joint)):
                SE_base_error2_joint.append((np.polyval(polynomials_base_error2_joint[index],
                    x_base_error2_joint[i]) - y_base_error2_joint[i])**2)



    ############################################################################

    # Compute the mean square error for each predictor
    MSE_sep = np.mean(SE_sep)
    MSE_joint = np.mean(SE_joint)

    # From the base error rates, compute an estimate of the difference between
    # joint and separate models given that the systems actually belong
    # to the same kind
    MSE_base_error1_sep = np.mean(SE_base_error1_sep)
    MSE_base_error1_joint = np.mean(SE_base_error1_joint)
    MSE_base_error2_sep = np.mean(SE_base_error2_sep)
    MSE_base_error2_joint = np.mean(SE_base_error2_joint)

    expected_difference = max(abs(MSE_base_error1_sep - MSE_base_error1_joint),
            abs(MSE_base_error2_sep - MSE_base_error2_joint))


    # FOR DEBUGGING ONLY
    print "MSE_sep = {0}, MSE_joint = {1}, MSE_base1 sep = {2}, MSE_base1 joint = {3}, MSE_base2 sep = {4}, MSE_base2 joint = {5}, expected difference = {6}".format(MSE_sep, MSE_joint, MSE_base_error1_sep, MSE_base_error1_joint, MSE_base_error2_sep, MSE_base_error2_joint, expected_difference)

    # Compare and return
    if MSE_joint > MSE_sep + alpha * expected_difference:
        return 1

    else:
        return 0
#        # Use full joint data set to fit new polynomials and return as a
#        # SymModel
#        data = np.concatenate((data1, data2))
#        
#        # Loop over curves in the data
#        cols = np.shape(data)[1]
#        for index in range(cols/2):
#            x_col = 2 * index
#            curve = data[:,x_col:(x_col+2)]
#            combined_sampled_data.append(curve)
#            combined_polynomials.append(FitPolyCV(curve, epsilon))
#
#        # Build and return the SymModel object
#        return SymModel(model1._index_var, model1._target_var,
#                model1._sys_id, combined_sampled_data,
#                combined_polynomials, epsilon)
