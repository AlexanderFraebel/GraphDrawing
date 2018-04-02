import string
import random
import numpy as np
from GraphNoAnim import *

def randAjdMat(N=5):
    R = np.zeros([N,N])
    for x in range(N):
        for y in range(N):
            R[x,y] = random.choice([1,1,0,0,0,0,0,0])
    return R

def checkCyclic(R):
    A = R.copy()
    for i in range(len(A)):
        A[i,i] = 0
    emptyRow = True
    while emptyRow == True:
        n = len(A)
        #print(n)
        if n == 0:
            #print(A)
            return False
        for i in range(n):
            if sum(A[i,]) == 0:
                A = np.delete(A,i,0)
                A = np.delete(A,i,1)
                emptyRow = True
                break
            emptyRow = False
    #print(A)
    return True


# Every acyclic graph has an adjacency matrix that can be rearranged so it is
# lower triangular. So to make an acyclic graph we just randomly fill in the
# lower triangular.
def makeAcyclic(N):
    R = np.zeros([N,N])
    for x in range(N):
        for y in range(x):
            R[x,y] = random.choice([1,1,0,0,0,0,0,0])
    return R

def connectogram(R,title=""):
    print(R)
    L = string.ascii_uppercase[:R.shape[0]]

    n = len(L)
    xy = arc((0,0),2.5,[0,np.pi*2],n)
    
    G = Graph(rdef=.3,tscaledef=70)
    
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str([L[i]][0]),z=2)
        
    G.QuickDraw()
    
    for i in np.argwhere(R==1):
        if i[0] == i[1]:
            continue
        A = G.Nodes[i[0]]
        B = G.Nodes[i[1]]
        ter = distpt(A,B,.32)
        connectArrPts(A.x,A.y,ter[0],ter[1],width=1.5,headwidth=.1,headlength=.2)

    plt.title(title)

#random.seed(100)

#L, R = randAjdMat(7)
#cyc = checkCyclic(R)
#if cyc:
#    t = "Cyclic"
#else:
#    t = "Acyclic"
#connectogram(L,R,title = "{} Graph".format(t))

R = makeAcyclic(9)
cyc = checkCyclic(R)
if cyc:
    t = "Cyclic"
else:
    t = "Acyclic"
connectogram(R,title = "{} Graph".format(t))