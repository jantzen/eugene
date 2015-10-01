'''
Created on Sep 30, 2015

@author: Colin
'''

# VABRangeDetermination.py
import numpy as np

def trisect(item):
    size = item.size
    firstInd = round(size / 3.0)
    first = item[0:firstInd]
    second = item[firstInd:firstInd * 2]
    third = item[firstInd * 2:size]
    return [first, second, third]

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
    best = [0, None]
    finished = False
    new = item
    
    newScore = score(new)
    
    if(newScore > best[0]):
        best = [newScore, new]
        print best
        
    while(not finished):
        current = trisect(new)
        scores = [0, 0, 0]
        curInd = None
        
        for i in range(0, 3):
            scores[i]=score(current[i])
            if(scores[i] > best[0]):
                best = [scores[i], current[i]]
                print best
                curInd=i
        if(curInd == None):
            finished = True
        else:
            if(curInd == 0):
                print scores
                if(aIsMoreLike(scores[0], scores[1], scores[2])=="B"):
                    new = np.concatenate((current[0], current[1]))
                else:
                    new = current[0]
            elif(curInd == 1):
                if(aIsMoreLike(scores[1], scores[0], scores[2])=="B"):
                    new = np.concatenate((current[0], current[1]))
                else:
                    new = np.concatenate((current[1], current[2]))
            else:
                if(aIsMoreLike(scores[2], scores[1], scores[0])=="B"):
                    new = np.concatenate((current[1], current[2]))
                else:
                    new = current[2]
            
            newScore = score(new)
            print new
            print newScore
            if(newScore > best[0]):
                best = [newScore, new]
                print best
                finished = False
                
    return best