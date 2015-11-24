import numpy as np
import scipy.stats as stats
import warnings

#Classes:
# SymModel

#Functions:
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

def BuildSymModel(data_frame, index_var, target_var, sys_id, epsilon=0):
    # from the raw curves of target vs. index, build tranformation 
    # curves (e.g., target1 vs. target2)
    abscissa = data_frame._target_values[0]
    ordinates = data_frame._target_values[1:]
    
    # for each curve, fit a polynomial using 10-fold cross-validation
    # to choose the order
#    pdb.set_trace()
    polynomials = []
    sampled_data = []

    for curve in ordinates:
        # first try a linear fit
        order = 1
        
        # partition the data
        data = np.vstack((abscissa, curve))
        data = data.transpose()
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

#        pdb.set_trace()
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
        polynomials.append(best_fit)

        sampled_data.append(data)

    # build and output a SymModel object
    return SymModel(index_var, target_var, sys_id, sampled_data, polynomials            , epsilon)


def CompareModels(model1, model2, delta=10**(-3)):
    """ Tests whether the models (and the systems they model) are equivalent. If
    so, it returns a combined model.
    """
#    pdb.set_trace()

    # Turn off warnings from polyfit
    warnings.simplefilter('ignore', np.RankWarning)


# FIND BEST FITS FOR JOINT MODEL
    
    # initialize containers for data that may be passed out
    combined_sampled_data = []
    combined_polynomials = []

    for counter, poly1 in enumerate(model1._polynomials):
        # import relevant data
        poly2 = model2._polynomials[counter]
        data1 = model1._sampled_data[counter]
        data2 = model2._sampled_data[counter]

        # fit the joint data
        data = np.vstack((data1,data2))
        combined_sampled_data.append(data)

        # first try a linear fit
        order = 1
        
        # partition the data
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
            if (mse - mse_candidate) / mse > min(model1._epsilon, model2._epsilon): 
                mse = mse_candidate
                best_fit_order = order
                best_fit = fit

            else:
                loop = False
            
#            # cap the complexity
#            if order >= 10:
#                loop = False

        # using the best-fit order, fit the full data set
        x = data[:,0]
        y = data[:,1]
        best_fit = np.polyfit(x, y, best_fit_order)
        combined_polynomials.append(best_fit)

        combined_sampled_data.append(data)

# USING BEST FITS FOR JOINT AND INDIVIDUAL MODELS, ESTIMATE ERROR

# INDIVIDUAL MODELS
    # stack the data for each separate model
    # x00 y00 x01 y01 x02 y02 ...
    # x10 y10 x11 y11 x12 y12 ...
    # ...
    # so all of the (xi0, yi0) describe a curve
    data1 = np.concatenate(model1._sampled_data, 1)
    data2 = np.concatenate(model2._sampled_data, 1)

    # partition the data
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    partition1 = np.array_split(data1, 10)
    partition2 = np.array_split(data2, 10)

    # compute the overall error using each partition as validation set
    square_errors = []
    for p in range(len(partition1)):
        # build training data
        s1 = np.shape(data1)[1]
        s2 = np.shape(data2)[1]
        training_set1 = np.empty([0,s1],dtype='float64')
        training_set2 = np.empty([0,s2],dtype='float64')
        for i in range(len(partition1)):
            if i != p:
                training_set1 = np.concatenate((training_set1, partition1[i]), 0)
                training_set2 = np.concatenate((training_set2, partition2[i]), 0)
        # loop over curves in the partition    
        for counter, poly1 in enumerate(model1._polynomials):
            # import relevant data
            poly2 = model2._polynomials[counter]

            # determine what order polynomial to fit
            order1 = len(poly1) - 1
            order2 = len(poly2) - 1

            # determine which columns of stacked data to use
            x_col = 2*(counter - 1)

            # fit polynomials
            x1 = training_set1[:,x_col]
            y1 = training_set1[:,(x_col+1)]
            fit1 = np.polyfit(x1, y1, order1)
  
            x2 = training_set2[:,x_col]
            y2 = training_set2[:,(x_col+1)]
            fit2 = np.polyfit(x2, y2, order2)
      
            # compute error of fit against partitition p
            x1 = partition1[p][:,x_col]
            y1 = partition1[p][:,(x_col+1)]
            x2 = partition2[p][:,x_col]
            y2 = partition2[p][:,(x_col+1)]
            for i in range(len(x1)):
                square_errors.append((np.polyval(fit1, x1[i]) - y1[i])**2)
            for i in range(len(x2)):
                square_errors.append((np.polyval(fit2, x2[i]) - y2[i])**2)
    # FOR DEBUGGING ONLY
    print "Length of square_errors, individual: {0}".format(len(square_errors))

    # compute mean square error for individual models 
    mse_ind = np.mean(np.array(square_errors))

# JOINT MODEL 
# stack the data 
    # x00 y00 x01 y01 x02 y02 ...
    # x10 y10 x11 y11 x12 y12 ...
    # ...
    # so all of the (xi0, yi0) describe a curve
    # data1 = np.concatenate(model1._sampled_data, 1)
    # data2 = np.concatenate(model2._sampled_data, 1)
    # data = np.concatenate((data1,data2))
    data = np.concatenate(combined_sampled_data, 1)

    # partition the data
    np.random.shuffle(data)
    partition = np.array_split(data, 10)

    # compute the overall error using each partition as validation set
    square_errors = []
    for p in range(len(partition)):
        # build training data
        s = np.shape(data)[1]
        training_set = np.empty([0,s],dtype='float64')
        for i in range(len(partition)):
            if i != p:
                training_set = np.concatenate((training_set, partition[i]), 0)
        # loop over curves in the partition    
        for counter, poly in enumerate(combined_polynomials):

            # determine what order polynomial to fit
            order = len(poly)-1

            # determine which columns of stacked data to use
            x_col = 2*(counter - 1)

            # fit polynomials
            x = training_set[:,x_col]
            y = training_set[:,(x_col+1)]
            fit = np.polyfit(x, y, order)
  
            # compute error of fit against partitition p
            x = partition[p][:,x_col]
            y = partition[p][:,(x_col+1)]
            for i in range(len(x)):
                square_errors.append((np.polyval(fit, x[i]) - y[i])**2)

# FOR DEBUGGING ONLY
    print "Length of square_errors, individual: {0}".format(len(square_errors))
               
    # compute mean square error for the joint model 
    mse_joint = np.mean(np.array(square_errors))

    # FOR DEBUGGING ONLY
    print "Individual mse: {0}, Joint mse: {1}".format(mse_ind, mse_joint)

    # if the mse of the individual models is not significantly less than that of the 
    # combined model, return the combined model; otherwise, return an empty list
   
    if mse_joint < mse_ind:
#    if (mse_ind - mse_joint)/mse_joint > delta:
        return SymModel(model1._index_var, model1._target_var,
                model1._sys_id, combined_sampled_data,
                combined_polynomials,min(model1._epsilon,
                    model2._epsilon))
    else:
        return None
