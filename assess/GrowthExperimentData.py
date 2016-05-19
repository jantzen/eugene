
from AccuracyAndRobustnessSuite import *
import pickle

# Run the noise experiment
print "Running the noise experiment."
[data_noise, CM_noise] = GrowthNoiseExperiment(25)

# Pickle the data
out = open('../outputs/NIPS2016/LG_data_noise.pkl','wb')
pickle.dump(data_noise, out)
out.close()
out = open('../outputs/NIPS2016/LG_CM_noise.pkl','wb')
pickle.dump(CM_noise, out)
out.close()


# Run the deviation from normality (skew) experiment
print "Running the deviation from normality experiment."
[data_skew, CM_skew] = GrowthSkewExperiment(25)

# Pickle the data
out = open('../outputs/NIPS2016/LG_data_skew.pkl','wb')
pickle.dump(data_skew, out)
out.close()
out = open('../outputs/NIPS2016/LG_CM_skew.pkl','wb')
pickle.dump(CM_skew, out)
out.close()


# Run the model comparison (different kind) experiment
print "Running the model discrimination experiment with variable K."
[data_model, CM_model] = GrowthExperimentDifferentKind(100)

# Pickle the data
out = open('../outputs/NIPS2016/LG_data_different.pkl','wb')
pickle.dump(data_model, out)
out.close()
out = open('../outputs/NIPS2016/LG_CM_different.pkl','wb')
pickle.dump(CM_model, out)
out.close()


# Run the model comparison (same kind) experiment
print "Running the model discrimination experiment with variable r."
[data_r, CM_r] = GrowthExperimentSameKind(100)

# Pickle the data
out = open('../outputs/NIPS2016/LG_data_same.pkl','wb')
pickle.dump(data_r, out)
out.close()
out = open('../outputs/NIPS2016/LG_CM_same.pkl','wb')
pickle.dump(CM_r, out)
out.close()

