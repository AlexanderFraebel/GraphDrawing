import random
import numpy as np
from Graphs import *

def randAjdMat(N=5):
    R = np.zeros([N,N])
    for x in range(N):
        for y in range(N):
            R[x,y] = random.choice([1,0,0,0,0,0,0,0,0])
    return R

# Doesn't work need to find a better method
def regularGraph(N=5,d=2,lim=100):
    ## Can't connect to more verticies than exist
    if d > (N-1):
        raise ValueError("Degree of a regular graph must be less than its size.")
    ## The number of vertices of odd degree must be even
    if N % 2 == 1 and d % 2 == 1:
        raise ValueError("No such graph due to handshaking lemma.")
    ctr = 0
    cr = False
    while cr == False:
        ctr += 1
        if ctr > lim:
            raise ValueError("Unable to find matching graph.")
        cr = True
        R = genmat(N,d)
        for i in R.values():
            if len(i) != d:
                cr = False
                break
    #print(ctr)
    return R
    
def genmat(N=5,d=2):
    S = [i for i in range(0,N)]
    D = {}
    for i in range(N):
        D[str(i)] = []
    for i in range(N):
        if len(S) == 0:
            break
        l = len(D[str(i)])
        if l < d:

            S = [x for x in S if x != i]
            n = min(d-l,len(S))
            r = random.sample(S,n)
            for v in r:
                D[str(i)].append(v)
                if i not in D[str(v)]:
                    D[str(v)].append(i)
            for pos, val in enumerate(S):
                if len(D[str(val)]) == d:
                    del S[pos]

    return D
            


def MatFromDict(D):
    R = np.zeros([len(D),len(D)])
    for key, value in D.items():
        R[int(key),value] = 1
    return R

N = 9
d = 2
D = regularGraph(N,d)
R = MatFromDict(D)
connectogramUndir(R,title="Degree {} Regular Graph".format(d),curve=1)

N = 7
d = 4
D = regularGraph(N,d)
R = MatFromDict(D)
connectogramUndir(R,title="Degree {} Regular Graph".format(d),curve=0,lineCol='green')