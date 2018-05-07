from Graphs import *
import math

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
