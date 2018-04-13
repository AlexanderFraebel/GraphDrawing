import random
import numpy as np
from Graphs import *


# Every acyclic graph has an adjacency matrix that can be rearranged so it is
# lower triangular. So to make an acyclic graph we just randomly fill in the
# lower triangular.
def makeAcyclic(N):
    R = np.zeros([N,N])
    for x in range(N):
        for y in range(x):
            R[x,y] = random.choice([1,1,0,0,0,0,0,0])
    return R


#random.seed(100)


R = makeAcyclic(9)
cyc = checkCyclic(R)
if cyc:
    t = "Cyclic"
else:
    t = "Acyclic"
G = connectogram(R,title = "{} Directed Graph".format(t))
