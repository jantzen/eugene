import numpy as np
import scipy.stats as stats
import warnings
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


###################################################################
###################################################################
#Functions



def FitPolyCV(data, epsilon=0):
    """ Takes a set of x,y data in the form of two columns (x in the first, y in the
        second) and fits a polynomial of order determined by 10-fold
        cross-validation
    """
    
    np.random.shuffle(data)
    partition = np.array_split(data, 10)

    # first try a linear fit
    order = 1
        
    # compute the error using each partition as validation set
    square_error = []
    for p in range(len(partition)):
        # build training data
        training_set = np.empty([0,2],dtype='float64')
        for i in range(len(partition)):
            if i != p:
                training_set = np.concatenate((training_set, partition[i]), 0)
        
        # fit polynomial
        x = training_set[:,0]
        y = training_set[:,1]
        fit = np.polyfit(x, y, 1)
        # compute error of fit against partitition p
        x = partition[p][:,0]
        y = partition[p][:,1]
        for i in range(len(x)):
            square_error.append((np.polyval(fit, x[i]) - y[i])**2)

    # compute mean square error for first order
    mse = np.mean(np.array(square_error))
    best_fit_order = order

    # assess fits for higher-order polynomials until the minimum is passed
    loop = True

    while loop:
        order += 1
        # partition the data
        # data = np.vstack((times, curve))
        # data = data.transpose()
        np.random.shuffle(data)
        partition = np.array_split(data, 10)

        # compute the error using each partition as validation set
        square_error = []
        for p in range(len(partition)):
           # build training data
           training_set = np.empty([0,2],dtype='float64')
           for i in range(len(partition)):
               if i != p:
                   training_set = np.concatenate((training_set, partition[i]), 0)
           # fit polynomial
           x = training_set[:,0]
           y = training_set[:,1]
           fit = np.polyfit(x, y, order)
           # compute error of fit against partition p
           x = partition[p][:,0]
           y = partition[p][:,1]
           for i in range(len(x)):
               square_error.append((np.polyval(fit, x[i]) - y[i])**2)

        # compute mean square error for current order
        mse_candidate = np.mean(np.array(square_error))

        # if significantly better, keep it. If not, keep the old and halt.
        if (mse - mse_candidate) / mse > epsilon:
            mse = mse_candidate
            best_fit_order = order
            best_fit = fit

        else:
            loop = False
            
        # shouldn't this cap be the only place where loop = Flase?
        # cap the complexity
        if order >= 10:
            loop = False

    # using the best-fit order, fit the full data set
    x = data[:,0]
    y = data[:,1]
    best_fit = np.polyfit(x, y, best_fit_order)

    return best_fit

 
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


def CompareModels(model1, model2, delta=10**(-3)):
    """ Tests whether the models (and the systems they model) are equivalent. If
        so, it returns a combined model.
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

    epsilon = min(model1._epsilon, model2._epsilon)

    # OUTER CROSS-VALIDATION LOOP
    ############################################################################
    
    # Partition the data
    partition1 = np.array_split(data1, 10)
    partition2 = np.array_split(data2, 10)

    # Prepare empty arrays to store squared errors for each test partition
    SE_sep = []
    SE_joint = []

    # Loop over partitions, using each as a test set once
    for p in range(10):
        # Build training data for each predictive model (separate vs. joint)
        cols = np.shape(partition1[0])[1]
        training_set_sep1 = np.empty([0, cols],dtype='float64')
        training_set_sep2 = np.empty([0, cols],dtype='float64')
        training_set_joint = np.empty([0, cols],dtype='float64')
        for i in range(10):
            if i != p:
                training_set_sep1 = np.concatenate((training_set_sep1,
                    partition1[i]), 0)
                training_set_sep2 = np.concatenate((training_set_sep2,
                    partition2[i]), 0)
                training_set_joint = np.concatenate((training_set_joint,
                    partition1[i], partition2[i]), 0)

        # TRAIN THE PREDICTORS
        ########################################################################
        polynomials_sep1 = []
        polynomials_sep2 = []
        polynomials_joint = []
        
        # Loop over curves in the training set
        for index in range(cols/2):
            x_col = 2 * index
            curve_sep1 = training_set_sep1[:,x_col:(x_col+2)]
            curve_sep2 = training_set_sep2[:,x_col:(x_col+2)]
            curve_joint = training_set_joint[:,x_col:(x_col+2)]

            polynomials_sep1.append(FitPolyCV(curve_sep1, epsilon))
            polynomials_sep2.append(FitPolyCV(curve_sep2, epsilon))
            polynomials_joint.append(FitPolyCV(curve_joint, epsilon))
            
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

            for i in range(len(x_sep1)):
                SE_sep.append((np.polyval(polynomials_sep1[index],
                        x_sep1[i]) - y_sep1[i])**2)

            for i in range(len(x_sep2)):
                SE_sep.append((np.polyval(polynomials_sep2[index],
                        x_sep2[i]) - y_sep2[i])**2)
            
            for i in range(len(x_joint)):
                SE_joint.append((np.polyval(polynomials_joint[index],
                    x_joint[i]) - y_joint[i])**2)


    ############################################################################

    # Compute the mean square error for each predictor
    MSE_sep = np.mean(SE_sep)
    MSE_joint = np.mean(SE_joint)

    # FOR DEBUGGING ONLY
    print "MSE_sep = {0}, MSE_joint = {1}".format(MSE_sep, MSE_joint)

    # Compare and return
    if MSE_joint > MSE_sep:
        return None

    else:
        # Use full joint data set to fit new polynomials and return as a
        # SymModel
        data = np.concatenate((data1, data2))
        
        # Loop over curves in the data
        cols = np.shape(data)[1]
        for index in range(cols/2):
            x_col = 2 * index
            curve = data[:,x_col:(x_col+2)]
            combined_sampled_data.append(curve)
            combined_polynomials.append(FitPolyCV(curve, epsilon))

        # Build and return the SymModel object
        return SymModel(model1._index_var, model1._target_var,
                model1._sys_id, combined_sampled_data,
                combined_polynomials, epsilon)
