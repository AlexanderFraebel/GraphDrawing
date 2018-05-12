from Graphs import *
from RandomPoints import randomNodes
from itertools import permutations
from time import time

def pathLength(L,G):
    out = 0
    t = G.Mat
    complete(G)
    D = DistanceMatrix(G)
    for i in range(len(L)-1):
        out += D[L[i],L[i+1]]
    G.Mat = t
    return out

# Brute Force
# Check every possible path until the shortest one is found
# Always gives the right answer but runs in factorial time so impossible for
# graphs of any significant sizr
def TravellingSalesmanPermutation(G):
    if G.size > 10:
        raise ValueError("Too Big for Brute Force Quickly")
    D = DistanceMatrix(G)
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

    return P
       
# Nearest Neighbor Search (Heuristic)
# Find the path going from nearest vertex to nearest vertex. This can vary
# depending on which vertex we start from so we start from each vertex. Time
# complexity is quadratic and results are usually close to the best path but
# are not guaranteed to be good.
def TravellingSalesmanNearest(G):
    D = DistanceMatrix(G)
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

def testTravellingSalesman():
    N = 10
    G1 = randomNodes(N,.1,NodeSize=.1,TextSize=1.5)#,seed=45345)
    complete(G1)
    t0 = time()
    path1 = TravellingSalesmanPermutation(G1)
    t1 = time()
    complete(G1)
    t2 = time()
    path2 = TravellingSalesmanNearest(G1)
    t3 = time()
    empty(G1)
    
    print("Brute Force Path in {:.5f} seconds".format(t1-t0))
    print(path1)
    print(pathLength(path1,G1))
    print("Best Nearest Neighbor Path in {:.5f} seconds".format(t3-t2))
    print(path2)
    print(pathLength(path2,G1))
    
    makeCanvas(size=[6,6])
    G1.addPath(path1)
    G1.drawNodes()
    G1.drawLines()
    
    makeCanvas(size=[6,6])
    empty(G)
    G1.addPath(path2)
    G1.drawNodes()
    G1.drawLines()
    
testTravellingSalesman()