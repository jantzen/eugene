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
#  exponents
#  npoly_val
#  residuals
#  surface_fit
#  FitPolyCV
#  BuildSymModel
#  CompareModels


###################################################################
###################################################################
#Classes


class SymModel( object ):
    """ An object for holding implicit numerical models of one ore more symmetry
        transformations. Each transformation is reprsented as a vector function
        of the target variables.

        index_var: the variable which functioned as the index for the sampled
        data; values of this variable are not preserved or represented in the
        symmetry model.

        target_vars: a list containing id numbers for each of the target
        variables measured and intervened upon

        sys_id: the id of the system from which the stored measurements were
        taken

        sampled_data: an (m-1)-element list of p x (n+1) arrays, where n is the
        number of target variables and p is the number of samples (length of
        array of index variable values)

        polynomials: an (m-1)-element list of n-element arrays, where m is the
        number of initial conditions sampled and n is the number
        of target variables.
    """
    
    def __init__(self, index_var, target_vars, sys_id, sampled_data, polynomials = [], epsilon=None):
        self._index_var = index_var
        self._target_vars = target_vars
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
    params_guess = np.zeros(num_params)

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
        if x_vars * order >= 10:
            loop = False
	
	#DEBUGGING
#	print "In PolyFitCV, current order = {}, mse = {}.".format(order, mse)

    # using the best-fit order, fit the full data set
    x = []
    for i in range(x_vars):
        x.append(data[:,i])
    y = data[:,x_vars]
    best_fit_params, cov = surface_fit(x, y, best_fit_order)

    return [best_fit_params, order]

 
def BuildSymModel(data_frame, index_var, target_vars, sys_id, epsilon=0):
    # from the raw blocks (Sets of curves of target vs. index variables), build transformation 
    # vector function (e.g., [target1', target2', target3'] = f([target1,
    # target2, target3]))
    abscissa = data_frame._target_values[0]
    ordinates = data_frame._target_values[1:]
    
    # for each block, fit a polynomial surface for each variable using 10-fold cross-validation
    # to choose the order (n n-variable polynomials for each of the m-1 blocks)
    polynomials = []
    sampled_data = []
    
    # DEBUGGING
    b = 0
    for block in ordinates:
        # extract the columns
        [num_rows, num_cols] = block.shape
        columns = []
        for i in range(num_cols):
            columns.append(block[:,i].reshape(num_rows,1))

        # format the data
        data = []
        for col in columns:
            data.append(np.hstack((abscissa, col)))

        # for each variable, fit a polynomial surface of the best order determined by 10-fold CV
        block_polys = []
	# DEBUGGING
	v = 0
        for d in data:
 	    #DEBUGGING
	    print "Currently working on sample {}, variable {}.".format(b, v)

            best_fit = FitPolyCV(d, epsilon)
            block_polys.append(best_fit)

	    v+=1

        # add to data arrays to pass out in final SymModel
        polynomials.append(block_polys)
        sampled_data.append(data)
    
	# DEBUGGING
	b += 1


    # build and output a SymModel object
    return SymModel(index_var, target_vars, sys_id, sampled_data, polynomials, 
            epsilon)


def CompareModels(model1, model2, alpha=1.):
    """ Tests whether the models (and the systems they model) are equivalent.
    Returns 0 if they are indistinguishable (of the same dynamical kind as far
    as we can tell) and 1 otherwise.
    """
    # determine values of some key parameters (for controlling loops)
    if len(model1._target_vars) == len(model2._target_vars):
        num_vars = len(model1._target_vars)
    else:
        raise ValueError('Models do not refer to the same number of target variables.')

    m = len(model1._sampled_data)

    # initialize containers for data that may be passed out
    combined_sampled_data = []
    combined_polynomials = []

    # PREPARE THE DATA
    rows1 = model1._sampled_data[0][0].shape[0]
    data1 = np.empty([rows1, 0],dtype='float64')
    rows2 = model2._sampled_data[0][0].shape[0]
    data2 = np.empty([rows2, 0],dtype='float64')

    # for each sampled block, randomize the data
    for sample in range(m):
        for block in model1._sampled_data[sample]:
            np.random.shuffle(block)
            data1 = np.hstack((data1, block))

        for block in model2._sampled_data[sample]:
            np.random.shuffle(block)
            data2 = np.hstack((data2, block))

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
                    base_error_partition1a[i]), 0)
                training_set_base_error1b = np.concatenate((training_set_base_error1b,
                    base_error_partition1b[i]), 0)
                training_set_base_error1_joint = np.concatenate((
                    training_set_base_error1_joint, base_error_partition1a[i], 
                    base_error_partition1b[i]), 0)
                training_set_base_error2a = np.concatenate((training_set_base_error2a,
                    base_error_partition2a[i]), 0)
                training_set_base_error2b = np.concatenate((training_set_base_error2b,
                    base_error_partition2b[i]), 0)
                training_set_base_error2_joint = np.concatenate((
                    training_set_base_error2_joint, base_error_partition2a[i], 
                    base_error_partition2b[i]), 0)
 

        # TRAIN THE PREDICTORS
        ########################################################################
        polynomials_sep1 = [[] for i in range(m)]
        polynomials_sep2 = [[] for i in range(m)]
        polynomials_joint = [[] for i in range(m)]

        polynomials_base_error1a = [[] for i in range(m)]
        polynomials_base_error1b = [[] for i in range(m)]
        polynomials_base_error1_joint = [[] for i in range(m)]
        polynomials_base_error2a = [[] for i in range(m)]
        polynomials_base_error2b = [[] for i in range(m)]
        polynomials_base_error2_joint = [[] for i in range(m)]
        
        # Loop over blocks in the training set
        # There should be (m-1) x n blocks, where m is number of initial
        # conditions sampled
        for i in range(m):
            for block in range(num_vars):
                block_start = m * block * (num_vars + 1)
                block_sep1 = training_set_sep1[:,block_start:(block_start + num_vars + 1)]
                block_sep2 = training_set_sep2[:,block_start:(block_start + num_vars + 1)]
                block_joint = training_set_joint[:,block_start:(block_start + num_vars + 1)]

                polynomials_sep1[i].append(FitPolyCV(block_sep1, epsilon))
                polynomials_sep2[i].append(FitPolyCV(block_sep2, epsilon))
                polynomials_joint[i].append(FitPolyCV(block_joint, epsilon))

                block_base_error1a = training_set_base_error1a[:,block_start:(block_start + num_vars 
                    + 1)]
                block_base_error1b = training_set_base_error1b[:,block_start:(block_start + num_vars 
                    + 1)]
                block_base_error1_joint = training_set_base_error1_joint[:,block_start:(block_start 
                    + num_vars + 1)]
                block_base_error2a = training_set_base_error2a[:,block_start:(block_start + num_vars 
                    + 1)]
                block_base_error2b = training_set_base_error2b[:,block_start:(block_start + num_vars 
                    + 1)]
                block_base_error2_joint = training_set_base_error2_joint[:,block_start:(block_start 
                    + num_vars + 1)]
 
                polynomials_base_error1a[i].append(FitPolyCV(block_base_error1a,
                    epsilon))
                polynomials_base_error1b[i].append(FitPolyCV(block_base_error1b,
                    epsilon))
                polynomials_base_error1_joint[i].append(FitPolyCV(block_base_error1_joint, 
                    epsilon))
                polynomials_base_error2a[i].append(FitPolyCV(block_base_error2a,
                    epsilon))
                polynomials_base_error2b[i].append(FitPolyCV(block_base_error2b,
                    epsilon))
                polynomials_base_error2_joint[i].append(FitPolyCV(block_base_error2_joint, 
                    epsilon))
 
            
        # TEST THE PREDICTORS
        ########################################################################
        # Loop over blocks in the test set (the index i ranges over blocks of
        # sampled data)
        for i in range(m):
            for block in range(num_vars):
                block_start = m * block * (num_vars + 1)

                x_sep1 = partition1[p][:,block_start:(block_start+num_vars)]
                y_sep1 = partition1[p][:,(block_start+num_vars+1)]
                x_sep2 = partition2[p][:,block_start:(block_start+num_vars)]
                y_sep2 = partition2[p][:,(block_start+num_vars+1)]
                x_joint = np.concatenate((partition1[p][:,block_start:(block_start+num_vars)], 
                    partition2[p][:,block_start:(block_start+num_vars)]))
                y_joint = np.concatenate((partition1[p][:,(block_start+num_vars+1)], 
                    partition2[p][:,(block_start+num_vars+1)]))

                x_base_error1a = base_error_partition1a[p][:,block_start:(block_start+num_vars)]
                y_base_error1a = base_error_partition1a[p][:,(block_start+num_vars+1)]
                x_base_error1b = base_error_partition1b[p][:,block_start:(block_start+num_vars)]
                y_base_error1b = base_error_partition1b[p][:,(block_start+num_vars+1)]
                x_base_error1_joint = np.concatenate((
                    base_error_partition1a[p][:,block_start:(block_start+num_vars)],
                    base_error_partition1b[p][:,block_start:(block_start+num_vars)]))
                y_base_error1_joint = np.concatenate((
                    base_error_partition1a[p][:,(block_start+num_vars+1)],
                    base_error_partition1b[p][:,(block_start+num_vars+1)]))

                x_base_error2a = base_error_partition2a[p][:,block_start:(block_start+num_vars)]
                y_base_error2a = base_error_partition2a[p][:,(block_start+num_vars+1)]
                x_base_error2b = base_error_partition2b[p][:,block_start:(block_start+num_vars)]
                y_base_error2b = base_error_partition2b[p][:,(block_start+num_vars+1)]
                x_base_error2_joint = np.concatenate((
                    base_error_partition2a[p][:,block_start:(block_start+num_vars)],
                    base_error_partition2b[p][:,block_start:(block_start+num_vars)]))
                y_base_error2_joint = np.concatenate((
                    base_error_partition2a[p][:,(block_start+num_vars+1)],
                    base_error_partition2b[p][:,(block_start+num_vars+1)]))
                
                # the index j runs over particular sample points
                for j in range(x_sep1.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_sep1[i][v][0],
                            exponents(num_vars,polynomials_sep1[i][v][1]), 
                            x_sep1[j,0:num_vars])
                        actual = y_sep1[j]
                        SE_sep.append(pow((predicted - actual),2))

                for j in range(x_sep2.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_sep2[i][v][0],
                            exponents(num_vars,polynomials_sep2[i][v][1]), 
                            x_sep2[j,0:num_vars])
                        actual = y_sep2[j]
                        SE_sep.append(pow((predicted - actual),2))

                for j in range(x_joint.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_joint[i][v][0],
                            exponents(num_vars,polynomials_joint[i][v][1]), 
                            x_joint[j,0:num_vars])
                        actual = y_joint[j]
                        SE_joint.append(pow((predicted - actual),2))

                for j in range(x_base_error1a.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_base_error1a[i][v][0],
                            exponents(num_vars,polynomials_base_error1a[i][v][1]), 
                            x_base_error1a[j,0:num_vars])
                        actual = y_base_error1a[j]
                        SE_base_error1_sep.append(pow((predicted - actual),2))

                for j in range(x_base_error1b.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_base_error1b[i][v][0],
                            exponents(num_vars,polynomials_base_error1b[i][v][1]), 
                            x_base_error1b[j,0:num_vars])
                        actual = y_base_error1b[j]
                        SE_base_error1_sep.append(pow((predicted - actual),2))

                for j in range(x_base_error1_joint.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_base_error1_joint[i][v][0],
                            exponents(num_vars,polynomials_base_error1_joint[i][v][1]), 
                            x_base_error1_joint[j,0:num_vars])
                        actual = y_base_error1_joint[j]
                        SE_base_error1_joint.append(pow((predicted - actual),2))

                for j in range(x_base_error2a.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_base_error2a[i][v][0],
                            exponents(num_vars,polynomials_base_error2a[i][v][1]), 
                            x_base_error2a[j,0:num_vars])
                        actual = y_base_error2a[j]
                        SE_base_error2_sep.append(pow((predicted - actual),2))

                for j in range(x_base_error2b.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_base_error2b[i][v][0],
                            exponents(num_vars,polynomials_base_error2b[i][v][1]), 
                            x_base_error2b[j,0:num_vars])
                        actual = y_base_error2b[j]
                        SE_base_error2_sep.append(pow((predicted - actual),2))

                for j in range(x_base_error2_joint.shape[0]):
                    for v in range(num_vars):
                        predicted = npoly_val(polynomials_base_error2_joint[i][v][0],
                            exponents(num_vars,polynomials_base_error2_joint[i][v][1]), 
                            x_base_error2_joint[j,0:num_vars])
                        actual = y_base_error2_joint[j]
                        SE_base_error2_joint.append(pow((predicted - actual),2))

#    pdb.set_trace()

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
    # return 1 if significantly different
    if MSE_joint > MSE_sep + alpha * expected_difference:
        return 1

    else:
        return 0
