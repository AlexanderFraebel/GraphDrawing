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
def randDegree(N=5,d=2):
    S = [i for i in range(0,N)]
    D = {}
    for i in range(N):
        D[str(i)] = []
    for i in range(N):
        if len(S) == 0:
            break
        #print(i)
        print(S)
        #print(D)
        #print()
        l = len(D[str(i)])
        if l < d and len(S) > (d-l):
            S = [x for x in S if x != i]
            r = random.sample(S,d-l)
            for v in r:
                D[str(i)].append(v)
                if i not in D[str(v)]:
                    D[str(v)].append(i)
            for pos, val in enumerate(S):
                if len(D[str(val)]) == d:
                    del S[pos]

        else:
            continue
    return D
            

def MatFromDict(D):
    R = np.zeros([len(D),len(D)])
    for key, value in D.items():
        R[int(key),value] = 1
    return R
        
D = randDegree(8,3)
R = MatFromDict(D)
print(D)
print(R)
connectogramUndir(R)