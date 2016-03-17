import eugene as eu
import AccuracyAndRobustnessSuite as AARS

import matplotlib.pyplot as plt
# import numpy as np
import pp

imageWidth = 10
imageHeight = 10
#facecolor = background color
#edgecolor = border color

def DisplayResults():
    sne = AARS.SimpleNoiseExperiment()
    noiselevels = sne.keys()
    scores = []
    for key in sne:
        scores.append(sne[key][2])


    snFig, snAx = plt.subplots(num="Simple Noise Experiment", 
                               figsize=(imageWidth, imageHeight))

    snAx.plot(noiselevels, scores)
    snFig.savefig('../assess/images/SimpleNoiseExperiment.png')
    snFig.show()


    
        
    #one graph for each **Experiment() ? 
################################################################3
################################################################3
DisplayResults()
