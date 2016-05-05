'''
Created on Sep 30, 2015

@author: Colin
'''

# VABRangeDetermination.py
import numpy as np
import scipy.signal
import math
import VABProcess as vp


global alpha, beta , gamma
alpha = 1
beta = 1
gamma = 1

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
        
def autoEmpD(array):
    if (array.size < 5):
        if (array.size <= 1):
            return 0
        return np.diff(array)
    
    if (array.size % 2) == 0:
        array = array[0:-1]
     
    deriv = np.zeros(array.size-5)   
    for i in range(0, array.size-5):
        sub = array[i:i+5]
        deriv[i] = vp.EmpiricalDeriv(sub)
    
    if deriv.size < 1:
        return 0
    return deriv

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
    
def curveFind2(item):    
    result = np.zeros(len(item))
    x1 = 0
    x2 = len(item)-1
    data = item - item[0]
    y1 = data[0]
    y2 = data[-1]
    ix = 0
    iy = 0
    
    if type(y1) is not np.float64:
        print y1
    
    magnitude = lambda a1, b1, a2, b2: np.sqrt(math.pow((a2-a1),2) + math.pow((b2-b1),2))
    
    mag = magnitude(x1, y1, x2, y2)
    if mag <= 0:
        # short segment
        return mag
    
    for i in range(0, data.size):
        px = i
        py = data[i]
        u = (((px - x1) * (x2-x1)) + ((py - y1) * (y2 - y1))) / mag
        if (u < 1*10**-5) or (u >= 1):
            ix = magnitude(px, py, x1, y1)
            iy = magnitude(px, py, x2, y2)
            s = (mag+ix+iy)/2
            
            z = (s*(s-mag)*(s-ix)*(s-iy))
            if z < 0:
                result[i] = 0
            else:
                result[i] = (2.0*np.sqrt(z))/mag
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            result[i] = magnitude(px, py, ix, iy)
        ix = 0
        iy = 0
            
    return result

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

def score2(array):
    
    score1 = 0
    score2 = 0
    score3 = 0
    # max-min seems very similar to using discrete differencing.
    # perhaps using the average value of the first difference is better?
    if array.size > 1:
        min1 = np.amin(array)
        max1 = np.amax(array)
        score1 = abs(max1-min1)/(math.sqrt(array.size))
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
                score3 = abs(max3-min3)/(math.sqrt(diff2.size))
    
    print score1
    print score2
    print "-------------------"
    return score1+score2
    
def score(array):
    sizeFactor = math.sqrt(array.size)
    score1 = 0
    score2 = 0
    if array.size > 5:
        emperical = autoEmpD(array)
        #print emperical
        score1 = np.sqrt(np.mean(np.square(autoEmpD(array))))
    if array.size > 3:
        score2 = np.average(curveFind2(array))
    
    print "Score 1: " + str(score1 * alpha)
    print "Score 2: " + str(score2 * beta)
    print "Size factor: " + str(1/(gamma * sizeFactor))
    
    score = (alpha*score1 + beta*score2)/(gamma * sizeFactor)
    
    print "Final Score: " + str(score)
    print "---------------------"
    return score        

def aIsMoreLike(a, b, c):
    if(abs(a-b)<abs(a-c)):
        return "B"
    else:
        return "C"
    
def findRange(item):
    # item is an array of format array([y0, y1, ..., yn])
    # best is a list of format [score, x-start, y-array]
    
    #smooth with scipy.signal.savgol_filter
    item = scipy.signal.savgol_filter(item, 5, 4)
    
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
    

def selectRange(samples, a=1, b=1, c=1):
    # samples is an nxm array, where n is the number of samples and m is the number of data points in each sample.
     # a, b and c will set the globals alpha, beta and gamma which change weights in the scoring function
    
    global alpha, beta, gamma
    alpha = a
    beta = b
    gamma = c
    
    
    n = len(samples)
    sects = np.zeros(n, dtype=sect)
    scores = np.zeros((n,n))
    i = 0
    
    for sample in samples:
        currentSect = findRange(sample)
        scores[i,i] = currentSect.score
        x1 = currentSect.start
        x2 = x1 + len(sample.data)
        sects[i] = currentSect
        j = 0
        

        for sample2 in samples:
            if i != j:
                newScore = score(sample2[x1:x2])
                scores[i,j] = newScore
            j += 1
        i += 1

    scorePerRange = np.sum(scores, axis = 1)
    k = 0
    best = 0
    bestInd = 0
    bestLen = 0
    
    for scored in scorePerRange:
        if (scored > best) or ((scored == best) and (len(sects[k].data) > bestLen)):
            bestInd = k
            best = scored
            bestLen = len(sects[k].data)
        k += 1
    
    print sects[bestInd]
    return sects[bestInd]

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
        