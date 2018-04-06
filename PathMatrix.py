from Graphs import *
import numpy as np
import random
import string

#random adjacency matrix
def randAjdMat(N=5):
    R = np.zeros([N,N])
    for x in range(N):
        for y in range(N):
            R[x,y] = random.choice([1,1,0,0,0,0,0,0,0,0])
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

# Convert an adjacency matrix into a path matrix
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

def connectogram(L,R):
    print(R)
    n = len(L)
    xy = arc((0,0),2.5,[0,np.pi*2],n)
    G = Graph(rdef=.3,tscaledef=70)
    
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str([L[i]][0]),z=2)
    
    G.Mat = R
    G.drawNodes()
    G.drawArrows()
    
    
def test():
    N = 5
    L = string.ascii_uppercase[:N]
    R = randAjdMat(N)
    P = pathMatrix(R)
    connectogram(L,R)
    connectogram(L,P)
test()