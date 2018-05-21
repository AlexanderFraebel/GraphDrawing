from Graphs import *
from RandomPoints import randomNodes

G = randomNodes(10,.5,NodeSize=.2,TextSize=1.5,seed=44345)
G.addEdges([8,3,4,1,1,7,9,7,2,2,9,2,4,7,8,2],
           [3,6,8,4,7,5,7,2,0,6,0,9,3,4,6,8],directed=True)

def mergeNode(a,b,G):

    A = [i for i in range(len(G.texts)) if G.texts[i] == str(a)][0]
    B = [i for i in range(len(G.texts)) if G.texts[i] == str(b)][0]

    x = np.mean([G.pos[A][0],G.pos[B][0]])
    y = np.mean([G.pos[A][1],G.pos[B][1]])
    
    G.Mat[A,:] += G.Mat[B,:]
    G.Mat[:,A] += G.Mat[:,B]
    
    G.delNode(B)
    
    G.pos[A] = [x,y]
    
    
def mergeNodes(L,G,name=None):
    L.sort()
    for i in range(len(L)-1):
        mergeNode(L[0],L[i+1],G)
    if name != None:
        G.texts[L[0]] = name
    print(G.texts)

makeCanvas()
G.drawNodes()
G.drawText()
G.drawArrows()


makeCanvas()
#mergeNodes([8,4,3,6],G)
mergeNodes([7,9,2],G)
G.drawNodes()
G.drawText()
G.drawArrows()

print(type(None))