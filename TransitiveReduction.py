import numpy as np
from Graphs import *

## I DON'T THINK THIS WORKS? ##


def makeAcyclic(N):
    R = np.zeros([N,N])
    for x in range(N):
        for y in range(x):
            R[x,y] = random.choice([1,1,0,0,0,0])
    return R

def booleanProduct(A,B):
    n = np.shape(A)[0]
    E = np.zeros(np.shape(A))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if A[i,k] == 1 and B[k,j] == 1:
                    E[i,j] = 1
                    break
    return E

def booleanPower(A,n=2):
    if n == 1:
        return A
    
    out = A.copy()
    for i in range(n-1):
        out = booleanProduct(out,A)
    return out

def pathMatrix(A):
    n = np.shape(A)[0]
    out = np.zeros(np.shape(A))
    for i in range(np.shape(A)[0]-1):
        out += booleanPower(A,i+1)

    for x in range(n):
        for y in range(n):
            if out[x,y] > 1:
                out[x,y] = 1

    return out


def transitiveReduce(P):
    t = P.copy()
    N = np.shape(R)[0]
    for j in range(N):
        for i in range(N):
            if t[i,j] == 1:
                for k in range(N):
                    if t[j,k] == 1:
                        t[i,k] = 0
    return t

N = 7
L = string.ascii_uppercase[:N]
A = makeAcyclic(N)
P = pathMatrix(A)
T = transitiveReduce(P)

connectogram(A,title="Adjacency",size=[4,4])
connectogram(P,title="Path",size=[4,4])
connectogram(T,title="Reduction",size=[4,4])