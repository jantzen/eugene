exec(open("alphaBetaGrid.py").read(), globals())
exec(open("LotkaVolterraSND.py").read(), globals())
exec(open("LVSim.py").read(), globals())
np.save("dataSpiderD1", resultsByPoint(list_of_points, 1, 500, 20))