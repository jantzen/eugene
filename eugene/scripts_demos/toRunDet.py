import numpy as np
exec(open("alphaBetaGrid.py").read(), globals())
exec(open("LotkaVolterraND.py").read(), globals())
exec(open("LVDSim.py").read(), globals())
np.save("spiderDataD2", resultsByPoint(list_of_points, 4000, 40))