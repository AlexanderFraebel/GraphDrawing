import Graphs
from RandomPoints import randomNodes
import random

def lineside(A,B,P):
    d = ( (P[0]-A[0])*(B[1]-A[1]) ) - ( (P[1]-A[1])*(B[0]-A[0]) )
    if d > 0:
        return True
    return False

def convexHull(G,directed=False):
    
    # Find the leftmost point and start from there
    xs = [i[0] for i in G.pos]
    cur = G.pos[np.argmin(xs)]
    
    # Keep track of positions and node labels
    P,O = [],[]
    ctr = 0
    while True:
        # Put the current point on the list
        P.append(cur)
        end = G.pos[0]
        
        mxj = 0
        # Any point could be next so we check that by going through them one
        # by one. If that point is to the left of the line (as seen from the
        # current point) then our current guess is wrong so we make that point
        # the next point and continue.
        for j in range(G.size):
            
            if cur == end or lineside(cur,end,G.pos[j]):
                mxj = j
                end = G.pos[j]
                
        O.append(mxj)
        # Once that is done we have our new current point
        cur = end
        
        
        # Draw the connections
        if len(P) > 1:
            G.addEdges([O[-1]],[O[-2]])
            if directed == False:
                G.addEdges([O[-2]],[O[-1]])
        
        # Check if we've reached the place where we started
        if end == P[0]:
            P.append(cur)
            G.addEdges([O[0]],[O[-1]])
            if directed == False:
                G.addEdges([O[-1]],[O[0]])
            break


    return O

makeCanvas()

#np.random.seed(17)
G = randomNodes(15,.3)
G.texts = [i for i in range(20)]
G.tscales = [2]*20
G.radii = [.15]*20
G.drawNodes()
G.drawText()

c = convexHull(G,directed=True)
print(c)
G.drawArrows()