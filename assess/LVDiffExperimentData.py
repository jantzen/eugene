# LotkaVolterraExperimentData.py

from AccuracyAndRobustnessSuite import *
import pickle


# Run the model comparison (different kind) experiment
[data_different, CM_different] = LVExperimentDifferentKind(100)

# Pickle the data
out = open('../outputs/NIPS2016/LV_data_different.pkl','wb')
pickle.dump(data_different, out)
out.close()
out = open('../outputs/NIPS2016/LV_CM_different.pkl','wb')
pickle.dump(CM_different, out)
out.close()
