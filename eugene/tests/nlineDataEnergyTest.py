from eugene.src.auxiliary.probability import *
import numpy as np

a = './durations_70_removed_sample.csv'
b = './durations_no_filter.csv'

def test_SAIDI_site_removal():

    def convertio(line):
        #print(line)
        try:
            q = float(line)
            return q 
        except:
            print('this is not a number', line)

    with open(a, 'r') as thefile:
        someFewer = [convertio(x) for x in thefile.read().split('\n') if x != '']
        someFewer = (np.asarray(someFewer)[np.newaxis]).T
        print(someFewer)

    with open(b, 'r') as thefile:
        noFewer = [convertio(x) for x in thefile.read().split('\n') if x != '']
        noFewer = (np.asarray(noFewer)[np.newaxis]).T

    someno = EnergyDistance(someFewer,noFewer)
    somesome = EnergyDistance(someFewer, someFewer)
    nono = EnergyDistance(noFewer, noFewer)

    assert someno > somesome
    assert someno > nono

    assert not significant(someFewer, someFewer, D=somesome, n=50)
    assert not significant(noFewer, noFewer, D=nono, n=50)

    # are noFewer and someFewer distributions significantly different?
    assert not significant(noFewer, someFewer, D=someno, n=50)
    # Good news!
