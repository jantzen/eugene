'''
Created on Sep 30, 2015

@author: Colin
'''

# VABRangeDetermination.py
import numpy as np

class sect (object):
    """ Class that represents a section of the original dataset.
    Contains a score, as set by the score() method,
    a starting position relative to the original dataset,
    and an array of values from the original dataset. 
    """
    def __init__(self, argData=np.array([]), argScore=None, argStart=None):
        self.data = argData
        self.score = argScore
        self.start = argStart
    
    def setScore(self, val):
        self.score = val
        
    def setData(self, array):
        self.data = array
        
    def setStart(self, val):
        self.start = val

def trisect(item):
    if item.data.size < 3:
        return [item, item, item]
        
    size = item.data.size
    firstInd = round(size / 3.0)
    first = item.data[0:firstInd]
    second = item.data[firstInd:firstInd * 2]
    third = item.data[firstInd * 2:size]
    start = item.start
    alpha = sect(first, score(first), start)
    beta = sect(second, score(second), start + firstInd)
    delta = sect(third, score(third), start + firstInd * 2)
    return [alpha, beta, delta]

def score(item):
    min = np.amin(item)
    max = np.amax(item)
    score = abs(max-min)/item.size
    return score

def aIsMoreLike(a, b, c):
    if(abs(a-b)<abs(a-c)):
        return "B"
    else:
        return "C"
    
def findRange(item):
    # item is an array of format array([y0, y1, ..., yn])
    # best is a list of format [score, x-start, y-array]
    best = sect()
    finished = False
    new = sect(item, score(item), 0)

    best = new
        
    while(not finished):
        current = trisect(new)
        scores = [0, 0, 0]
        curInd = None
        
        for i in range(0, 3):
            scores[i]=current[i].score
            if(scores[i] > best.score):
                best = current[i]
                curInd=i
        if(curInd == None):
            finished = True
        else:
            startx = 0
            if(curInd == 0):
                if(aIsMoreLike(scores[0], scores[1], scores[2])=="B"):
                    newData = np.concatenate((current[0].data, current[1].data))
                    startx = current[0].start
                else:
                    newData = current[0].data
                    startx = current[0].start
            elif(curInd == 1):
                if(aIsMoreLike(scores[1], scores[0], scores[2])=="B"):
                    newData = np.concatenate((current[0].data, current[1].data))
                    startx = current[0].start
                else:
                    newData = np.concatenate((current[1].data, current[2].data))
                    startx = current[1].start
            else:
                if(aIsMoreLike(scores[2], scores[1], scores[0])=="B"):
                    newData = np.concatenate((current[1].data, current[2].data))
                    startx = current[1].start
                else:
                    newData = current[2].data
                    startx = current[2].start
            
            new = sect(newData, score(newData), startx)
            if(new.score > best.score):
                best = new
                finished = False
                
    return best