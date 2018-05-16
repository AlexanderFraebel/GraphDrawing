from Graphs import *
from RandomPoints import randomNodes
from itertools import permutations
from time import time

def pathLength(L,G):
    out = 0
    D = G.Dist
    for i in range(len(L)-1):
        out += D[L[i],L[i+1]]
    return out



# Brute Force
# Check every possible path until the shortest one is found
# Always gives the right answer but runs in factorial time so impossible for
# graphs of any significant sizr
def TSPPermutation(G):
    if G.size > 10:
        raise ValueError("Too Big for Brute Force Quickly")
    D = maskDist(G)
    l = [i for i in range(G.size)]
    DS = np.inf
    P = []
    for i in permutations(l):
        ds = 0
        for x in range(G.size-1):
            ds += D[i[x],i[x+1]]
        if ds < DS:
            DS = ds
            P = i

    return list(P)
       
# Nearest Neighbor Search (Heuristic)
# Find the path going from nearest vertex to nearest vertex. This can vary
# depending on which vertex we start from so we start from each vertex. Time
# complexity is quadratic and results are usually close to the best path but
# are not guaranteed to be good.
def TSPNearest(G):
    D = maskDist(G)    
    final = []
    dists = [0]*G.size
    for start in range(G.size):
        S = [i for i in range(G.size)]
        out = [start]
        del S[start]
        for i in range(G.size-1):
            cur = out[-1]
            mn = np.inf
            for p,j in enumerate(S):
                if D[cur,j] < mn:
                    mn = D[cur,j]
                    ps = p
            dists[start] += mn
            out.append(S.pop(ps))
        final.append(out)
    return final[np.argmin(dists)]

# Held-Karp algorithm produces a tour of the cities much more efficiently than
# a brute force search.
def TSPHeldKarp(G):
    p = dict()
    C = dict()
    D = maskDist(G)  
    hkfun(0,[i for i in range(1,G.size)],D,p,C)
    #print(p)
    #print(len(p))
    return ptrace(p,G.size)

def hkfun(v,L,D,p,C):
    if len(L) == 1:
        s = str(v) + str(L)
        p[s] = v
        return D[v,L] + D[L,0]
    if str(v) + str(L) in p.keys():
        return C[str(v) + str(L)]
    else:
        out = []
        T = []
        for pos,val in enumerate(L):
            tL = L.copy()
            del tL[pos]
            d = D[v,val] + hkfun(val,tL,D,p,C)
            out.append(d)
            T.append(val)
        s = str(v) + str(L)
        p[s] = T[np.argmin(out)]
        C[s] = min(out)
        return min(out)

def ptrace(p,N):
    out = [0]
    L = [i for i in range(1,N)]
    # Find the first section
    s = "0"
    s += str(L)
    curV = p[s]
    L.remove(curV)
    out.append(curV)

    for i in range(N-3):
        s = str(curV)
        s += str(L)
        curV = p[s]
        L.remove(curV)
        out.append(curV)
    out.append(L[0])
    out.append(0)
    return out
        




def testTravellingSalesman():
    N = 14
    G1 = randomNodes(N,.2,NodeSize=.1,TextSize=1.5)#,seed=45345)
    complete(G1)
    #t0 = time()
    #path1 = TSPPermutation(G1)
    #t1 = time()
    t2 = time()
    path2 = TSPNearest(G1)
    t3 = time()
    t4 = time()
    path3 = TSPHeldKarp(G1)
    t5 = time()
    empty(G1)
    
    #print("Brute Force Path in {:.2f} seconds".format(t1-t0))
    #print(path1)
    #print("{:.3f}\n".format(pathLength(path1,G1)))
    print("Best Nearest Neighbor Path in {:.2f} seconds".format(t3-t2))
    print(path2)
    print("{:.3f}\n".format(pathLength(path2,G1)))
    print("Hale-Karp Cycle in {:.2f} seconds".format(t5-t4))
    print(path3)
    print("{:.3f}".format(pathLength(path3,G1)))

    
    #makeCanvas(size=[4,4])
    #G1.addPath(path1)
    #G1.drawNodes()
    #G1.drawLines()
    
    makeCanvas(size=[4,4])
    empty(G1)
    G1.addPath(path2)
    G1.drawNodes()
    G1.drawLines()
    
    makeCanvas(size=[4,4])
    empty(G1)
    G1.addPath(path3)
    G1.drawNodes()
    G1.drawLines()
    
    print("\n")
    
testTravellingSalesman()

