
from AccuracyAndRobustnessSuite import *
import pickle

# Run the noise experiment
[data_noise, CM_noise] = CircuitNoiseExperiment(20)

# Pickle the data
out = open('../outputs/data_noise_circ.pkl','wb')
pickle.dump(data_noise, out)
out.close()
out = open('../outputs/CM_noise_circ.pkl','wb')
pickle.dump(CM_noise, out)
out.close()
