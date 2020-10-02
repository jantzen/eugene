from itertools import combinations
from functools import reduce

def sigma2(X, Y):
    # list comprehension style double sigma
    distances = sum([abs(x - y) for x in X for y in Y])
    return distances

def twoSample(X, Y):
    """
    Eq 5 page 5 Szekely and Rizzo, Testing for Equal
    Distribution in High Dimension
    """
    n1 = len(X)
    n2 = len(Y)
    mult = (n1 * n2) / (n1 + n2)
    xy = sigma2(X,Y)
    xx = sigma2(X,X)
    yy = sigma2(Y,Y)
    return mult * (2/(n1*n2) * xy - 1/n1**2 * xx - 1/n2**2 * yy)

def kSample(*args):
    col1 = []
    col2 = []
    pairs = combinations(args,2)
    return reduce(lambda a,b: a+b, [(twoSample(tup[0],tup[1])) for tup in pairs])


