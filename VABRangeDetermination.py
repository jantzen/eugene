'''
Created on Sep 30, 2015

@author: Colin
'''

# VABRangeDetermination.py
import numpy as np
import math
import VABProcess as vp

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

def curveFind(item):
    data = np.zeros(item.size)
    rise = item[-1]-item[0]
    run = item.size - 1
    
    line = lambda x: (rise/run)*x+item[0]
    
    i = 0
    for datum in item:
        data[i] = datum - line(i)
        i = i + 1
        
    return data

def trisect(item):
    if item.data.size < 3:
        # we can't split an atom, so return the item in triplicate.
        # this forces the algorithm into stopping.
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

def score(array):
    
    score1 = 0
    score2 = 0
    score3 = 0
    # max-min seems very similar to using discrete differencing.
    # perhaps using the average value of the first difference is better?
    if array.size > 1:
        min1 = np.amin(array)
        max1 = np.amax(array)
        score1 = abs(max1-min1)/(array.size/math.sqrt(array.size))
        diff1 = np.diff(array)
    
        if diff1.size > 1:
            min2 = np.amin(diff1)
            max2 = np.amax(diff1)
            score2 = np.average(diff1)
            #score2 = abs(max2-min2)/(diff1.size/math.sqrt(diff1.size))
            diff2 = np.diff(array, n=2)
    
            if diff2.size > 1:
                min3 = np.amin(diff2)
                max3 = np.amax(diff2)
                score3 = abs(max3-min3)/(diff2.size/math.sqrt(diff2.size))
    
    print score1
    print score2
    print "-------------------"
    return score1+score2

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

def findMonotone(item, start = 0):
    # item is an array of format array([y0, y1, ..., yn])
    index = start
    previous = item[index]
    climb = -1
    
    for datum in item[index+1:]:
        index = index + 1
        if datum > previous:
            if index == start + 1:
                # this is our second piece of data, we're climbing
                climb = True
                # look for a max
            elif climb == False:
                return index
        elif datum < previous:
            if index == start + 1:
                # this is our second piece of data, we're climbing
                climb = False
                # look for a min
            elif climb == True:
                return index
        previous = datum
        