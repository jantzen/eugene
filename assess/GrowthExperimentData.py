
from AccuracyAndRobustnessSuite import *
import pickle

# Run the noise experiment
print "Running the noise experiment."
[data_noise, CM_noise] = GrowthNoiseExperiment(25)

# Pickle the data
out = open('../outputs/data_noise.pkl','wb')
pickle.dump(data_noise, out)
out.close()
out = open('../outputs/CM_noise.pkl','wb')
pickle.dump(CM_noise, out)
out.close()


# Run the deviation from normality experiment
print "Running the deviation from normality experiment."
[data_skew, CM_skew] = GrowthNormDeviationExperiment(25)

# Pickle the data
out = open('../outputs/data_skew.pkl','wb')
pickle.dump(data_skew, out)
out.close()
out = open('../outputs/CM_skew.pkl','wb')
pickle.dump(CM_skew, out)
out.close()


# Run the model discrimination experiment (K varies)
print "Running the model discrimination experiment with variable K."
[data_model, CM_model] = GrowthModelExperiment(25)

# Pickle the data
out = open('../outputs/data_model.pkl','wb')
pickle.dump(data_model, out)
out.close()
out = open('../outputs/CM_model.pkl','wb')
pickle.dump(CM_model, out)
out.close()


# Run the model discrimination experiment (r varies)
print "Running the model discrimination experiment with variable r."
[data_r, CM_r] = GrowthRateExperiment(25)

# Pickle the data
out = open('../outputs/data_r.pkl','wb')
pickle.dump(data_r, out)
out.close()
out = open('../outputs/CM_r.pkl','wb')
pickle.dump(CM_r, out)
out.close()


