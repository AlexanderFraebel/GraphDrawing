from Graphs import *
import math

###############################################################################
##
## For defining the graph itself
##
###############################################################################


def randAjdMat(N=5):
    R = np.zeros([N,N],dtype="int")
    for x in range(N):
        for y in range(N):
            R[x,y] = random.choice([1,1,0,0,0,0,0])
    return R

###############################################################################
##
## IMPLEMENTING THE ALGORITHM
##
###############################################################################



def updatedists(x,M,d,pred,visited,unvisited):
    # Check each neighbot of node x
    for num,val in enumerate(M[x,]):
        # If it is unvisited
        if num not in visited and val != 0:
            # Check if going to it from this node is the shortest known path
            if d[x] + M[x,num] < d[num]:
                # If it is record the distance and set this node as its predecessor
                d[num] = d[x] + M[x,num]
                pred[num] = x
    # Set this node as visited
    unvisited[x] = 0
    visited.append(x)

    # Choose the next node
    t = math.inf
    out = 0
    for i in unvisited:
        if i != 0 and d[i] <= t:
            t = d[i]
            out = i
    return out

## Wrapper function
def dijkstra(M,s=0):

    ## List of distances from start are all unknown to begin with
    d = [math.inf]*len(M)

    ## Distance from start to itself is zero
    d[s] = 0

    ## Unvisited nodes
    unvisited = [i for i in range(len(M))]
    ## Visited nodes
    visited = []
    ## The predecessor of each element along the shortest path
    pred = [s]*len(M)

    # We start from node 0 when we run the updatedists function it changes the
    # mutable objects and then returns the next node to visit
    cur = s
    for i in range(len(M)):
        cur = updatedists(cur,M,d,pred,visited,unvisited)
    
    # Create a list of shortest paths from the start node s to each other node
    # If non-existent the path is an empty list
    pathlist = []
    for i in range(len(pred)):
        if d[i] == math.inf:
            pathlist.append([])
            continue
        path = []
        t = i
        while t != s:
            path.insert(0,t)
            t = pred[t]
        path.insert(0,s)
        pathlist.append(path)

    return d,pathlist

def DijkstraExample():
    X = randAjdMat(9)
    #print(X)
    N = 0
    di = dijkstra(X,N)
    G = connectogram(X,[i for i in range(8)],
                       title="Shortest Paths from Node {}".format(N))
    #print(di[0])
    for pos,val in enumerate(di[1]):
        s = ""
        for i in val:
            s += str(i) + " \u2192 "
        print("{}: {}".format(pos,s[:-2]))
        
    for i in di[1]:
        if len(i) == 1:
            continue
        for j in range(len(i)-1):
            connectArr(G.Nodes[i[j]],G.Nodes[i[j+1]],col='red',z=1,width=2)


DijkstraExample()