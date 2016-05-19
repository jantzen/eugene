# LotkaVolterraExperimentData.py

from AccuracyAndRobustnessSuite import *
import pickle

# Run the noise experiment
[data_noise, CM_noise] = LVNoiseExperiment(25)

# Pickle the data
out = open('../outputs/NIPS2016/LV_data_noise.pkl','wb')
pickle.dump(data_noise, out)
out.close()
out = open('../outputs/NIPS2016/LV_CM_noise.pkl','wb')
pickle.dump(CM_noise, out)
out.close()


# Run the deviation from normality (skew) experiment
[data_skew, CM_skew] = LVSkewExperiment(25)

# Pickle the data
out = open('../outputs/NIPS2016/LV_data_skew.pkl','wb')
pickle.dump(data_skew, out)
out.close()
out = open('../outputs/NIPS2016/LV_CM_skew.pkl','wb')
pickle.dump(CM_skew, out)
out.close()


# Run the model comparison (different kind) experiment
[data_different, CM_different] = LVExperimentDifferentKind(100)

# Pickle the data
out = open('../outputs/NIPS2016/LV_data_different.pkl','wb')
pickle.dump(data_different, out)
out.close()
out = open('../outputs/NIPS2016/LV_CM_different.pkl','wb')
pickle.dump(CM_different, out)
out.close()


# Run the model comparison (same kind) experiment
[data_same, CM_same] = LVExperimentSameKind(100)

# Pickle the data
out = open('../outputs/NIPS2016/LV_data_same.pkl','wb')
pickle.dump(data_same, out)
out.close()
out = open('../outputs/NIPS2016/LV_CM_same.pkl','wb')
pickle.dump(CM_same, out)
out.close()
