import random
import numpy as np
from Graphs import *

def randAjdMat(N=5):
    R = np.zeros([N,N])
    for x in range(N):
        for y in range(N):
            R[x,y] = random.choice([1,0,0,0,0,0,0,0,0])
    return R


            



N = 9
d = 2
R = regularGraph(N,d)
connectogramUndir(R,title="Degree {} Regular Graph".format(d),curve=1)

N = 7
d = 4
R = regularGraph(N,d)
connectogramUndir(R,title="Degree {} Regular Graph".format(d),curve=0,lineCol='green')