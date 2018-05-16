from Graphs import *
from RandomPoints import randomNodes
from time import time

# Create a list of the edges in the Graph sortest from shortest to longest
def distList(G):
    L = []
    E = []
    for i in range(G.size):
        for j in range(i):
            if G.Mat[i,j] != 0 or G.Mat[j,i] != 0:
                L.append(G.Dist[i,j])
                E.append([i,j])

    o = np.argsort(L)
    E = np.asarray(E)
    return np.ndarray.tolist(E[o])
            
            
## Prim's algorithm grows the tree edge by edge always adding the point that is
## closest to the portion of the tree that has been built
def PrimMinTree(G,modify=True):
    E = distList(G)
    if modify == True:
        empty(G)
    T = set()
    T.add(0)
    out = []
    while len(T) < G.size:
        for edge in E:
            if (edge[0] not in T and edge[1] in T) or (edge[1] not in T and edge[0] in T):
                
                T.add(edge[0])
                T.add(edge[1])
                out.append(edge)
                if modify == True:
                    G.addEdges([edge[0]],[edge[1]])
                break
    return out

## Kruskal's algorithm is more efficient, though more complex. It first 
## measures each edge and then sorts them. Going through the list in order an 
## edge is added only if it connects two components that aren't already 
## connected.
def KruskalMinTree(G,modify=True):
    # Create a list of all valid edges sorted by length
    E = distList(G)
    if modify == True:
        empty(G)
    # Initially every node is its own tree in the forest
    F = [[i] for i in range(G.size)]
    out = []
    for edge in E:
        # Find which tree each end of the edge is in
        p1 = 0
        p2 = 0
        for ps,f in enumerate(F):
            if edge[0] in f:
                p1 = ps
            if edge[1] in f:
                p2 = ps
        # If they're in the same tree go on to the next edge
        if p1 == p2:
            continue
        # Otherwise merge the two trees and put the edge onto the list
        F[p1] += F[p2]
        del F[p2]
        out.append(edge)
        if modify == True:
            G.addEdges([edge[0]],[edge[1]])
            
        # No need to keep going through the list if all the trees have been
        # merged already.
        if len(F) == 1:
            break

    return out


def testMinimumSpanningTree():
    sd = np.random.randint(1,999)
    N = 150
    
    makeCanvas(size=[7,7])
    G = randomNodes(N,.1,NodeColor='black',NodeSize=.05,seed=sd)
    complete(G)
    
    t0 = time()
    Krus = KruskalMinTree(G,modify=False)
    print("Kruskal: {:.3f} seconds".format(time()-t0))
    #print(Krus)
    
    empty(G)
    G.addEdges(Krus)
    G.drawNodes()
    G.drawLines()
    
    
    
    makeCanvas(size=[7,7])
    G = randomNodes(N,.1,NodeColor='black',NodeSize=.05,seed=sd)
    complete(G)
    
    t0 = time()
    Prim = PrimMinTree(G,modify=False)
    print("Prim:    {:.3f} seconds".format(time()-t0))
    #print(Prim)
    
    empty(G)
    G.addEdges(Prim)
    G.drawNodes()
    G.drawLines()
    
#testMinimumSpanningTree()