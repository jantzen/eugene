
from AccuracyAndRobustnessSuite import *
import pickle

# Run the noise experiment
[data_noise, CM_noise] = GrowthNoiseExperiment(1)

# Pickle the data
out = open('../outputs/data_noise.pkl','wb')
pickle.dump(data_noise, out)
out.close()
out = open('../outputs/CM_noise.pkl','wb')
pickle.dump(CM_noise, out)
out.close()

# Run the deviation from normality experiment
[data_skew, CM_skew] = GrowthNormDeviationExperiment(1)

# Pickle the data
out = open('../outputs/data_skew.pkl','wb')
pickle.dump(data_skew, out)
out.close()
out = open('../outputs/CM_skew.pkl','wb')
pickle.dump(CM_skew, out)
out.close()


# Run the model discrimination experiment
[data_model, CM_model] = GrowthModelExperiment(1)

# Pickle the data
out = open('../outputs/data_model.pkl','wb')
pickle.dump(data_model, out)
out.close()
out = open('../outputs/CM_model.pkl','wb')
pickle.dump(CM_model, out)
out.close()


