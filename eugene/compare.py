import numpy as np
import scipy.stats as stats

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
    
    def __init__(self, index_var, target_var, sys_id, sampled_data, polynomials
            = [], sse=None, NumberOfSamples=0, epsilon=None):
        self._index_var = index_var
        self._target_var = target_var
        self._sys_id = sys_id
        self._sampled_data = sampled_data
        self._polynomials = polynomials
        self._sse = sse
        self._NumberOfSamples = NumberOfSamples
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
    SSerrors = []
    sampled_data = []
    NumberOfSamples = 0

    for curve in ordinates:
        # first try a linear fit
        order = 1
        
        # partition the data
        data = np.vstack((abscissa, curve))
        data = data.transpose()
        np.random.shuffle(data)
        partition = np.array_split(data, 10)

        # compute the error using each partition as validation set
        mse = []
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
            sum_square_error = 0
            for i in range(len(x)):
                sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
            mse.append(sum_square_error/len(x))

        # compute mean square error for first order
        mmse = np.mean(np.array(mse))
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
            mse = []
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
               sum_square_error = 0
               for i in range(len(x)):
                   sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
               mse.append(sum_square_error/len(x))

            # compute mean square error for current order
            mmse_candidate = np.mean(np.array(mse))

            # if significantly better, keep it. If not, keep the old and halt.
            if (mmse - mmse_candidate) / mmse > epsilon:
                mmse = mmse_candidate
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

        # compute the sum of squared errors for the best fit and append to the
        # list
        sse = 0
        for i in range(len(x)):
            sse += (np.polyval(best_fit, x[i]) - y[i])**2
        SSerrors.append(sse)
        
        sampled_data.append(data)
        NumberOfSamples += len(x)
    
    # build and output a SymModel object
    sse = sum(SSerrors)
    return SymModel(index_var, target_var, sys_id, sampled_data, polynomials,
            sse, NumberOfSamples, epsilon)


def CompareModels(model1, model2):
    """ Tests whether the models (and the systems they model) are equivalent. If
    so, it returns a combined model.
    """
#    pdb.set_trace()
    p_vals = []

    # initialize containers for data that may be passed out
    combined_sampled_data = []
    combined_polynomials = []
    SSerrors = []
    NumberOfSamples = 0

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
        mse = []
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
            sum_square_error = 0
            for i in range(len(x)):
                sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
            mse.append(sum_square_error/len(x))

        # compute mean square error for first order
        mmse = np.mean(np.array(mse))
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
            mse = []
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
               sum_square_error = 0
               for i in range(len(x)):
                   sum_square_error += (np.polyval(fit, x[i]) - y[i])**2
               mse.append(sum_square_error/len(x))

            # compute mean square error for current order
            mmse_candidate = np.mean(np.array(mse))

            # if significantly better, keep it. If not, keep the old and halt.
            if (mmse - mmse_candidate) / mmse > min(model1._epsilon, model2._epsilon): 
                mmse = mmse_candidate
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

        # compute the sum of squared errors for the best fit and append to the
        # list
        sse = 0
        for i in range(len(x)):
            sse += (np.polyval(best_fit, x[i]) - y[i])**2
        SSerrors.append(sse)
        
        combined_sampled_data.append(data)
        NumberOfSamples += len(x)
 
#        null_hyp = np.polyfit(x, y, best_fit_order)
#
#        # save the best fit polynomial
#        combined_polynomials.append(null_hyp)
#
#        # now use the same 'model' (the same order polynomial) to fit each data
#        # set individually
#        poly1 = np.polyfit(data1[:,0], data1[:,1], best_fit_order)
#        poly2 = np.polyfit(data2[:,0], data2[:,1], best_fit_order)
#
#        # compute sum of squares (SS) and degrees of freedom (df) for the null
#        # hypothesis: both data sets are described by the same polynomial
#
#        # first, construct a function to norm the data
#        norm = lambda x: (x - np.mean(data[:,1])) / (np.max(data[:,1]) -
#            np.min(data[:,1])) 
#
#        # SS null
#        SSnull = 0
#        for i in range(len(data)):
#            x = data[i,0]
#            y = data[i,1]
#            SSnull += np.power(norm(y) - norm(np.polyval(null_hyp, x)), 2.)
#
#        # df null
#        df_null = len(data) - len(null_hyp)
#
#        # SS alt
#        SSalt = 0
#        for i in range(len(data1)):
#            x = data1[i, 0]
#            y = data1[i, 1]
#            SSalt += np.power(norm(y) - norm(np.polyval(poly1, x)), 2.)
#
#        for i in range(len(data2)):
#            x = data2[i, 0]
#            y = data2[i, 1]
#            SSalt += np.power(norm(y) - norm(np.polyval(poly2, x)), 2.)
#        
#        # df alt
#        df_alt = len(data1) - len(poly1) + len(data2) - len(poly2)
#
#        # F-statistic
#        F = ((SSnull - SSalt)/SSalt)/((float(df_null) -
#            float(df_alt))/float(df_alt))
#
#        # p-value
#        p = 1. - stats.f.cdf(F, (df_null - df_alt), df_alt) 
#
#        p_vals.append(p)

#    pdb.set_trace()

    # if most of the p_vals exceed alpha = 0.05, then conclude that the models
    # are equivalent and return the new combined model; otherwise, return an
    # empty list. 
#    p_vals = np.array(p_vals)
#    if (np.sum(np.greater(p_vals, np.ones(p_vals.shape)*0.05)) >
#         round(len(p_vals)/2.)):
#        return SymModel(model1._index_var, model1._target_var, model1._sys_id,
#                combined_sampled_data, combined_polynomials,
#                min(model1._epsilon, model2._epsilon))
#    else:
#        return None

    # if the mse of the individual models is not significantly less than that of the 
    # combined model, return the combined model; otherwise, return an empty list
    mse_individual = (model1._sse + model2._sse)/(model1._NumberOfSamples +
            model2._NumberOfSamples)
    mse_combined = sum(SSerrors)/NumberOfSamples
    if (mse_individual - mse_combined)/mse_combined > min(model1._epsilon,
            model2._epsilon):
        return SymModel(model1._index_var, model1._target_var,
                model1._sys_id, combined_sampled_data,
                combined_polynomials,sum(SSerrors),NumberOfSamples,min(model1._epsilon,
                    model2._epsilon))
    else:
        return None
