from Graphs import *
from ParticularGraphs import completeBipartiteGraph

def subgraphDeck(G):
    out = []
    for i in range(G.size):
        L = [x for x in range(G.size)]
        del L[i]
        out.append(subgraph(G,L))
        
    return out


makeCanvas()
G = Graph(NodeSize=.3)
G.addNodes([[0,0],[.5,1],[-1,-1],[-1.3,.7],[0,2],[-2.2,-.7]])
G.addEdges([0,1,2,1,2,4,5,5],
           [3,0,0,3,3,0,1,2])
G.drawNodes()
G.drawLines()
G.drawText()

makeCanvas(size=[15,15])
T = subgraphDeck(G)
for pos,i in enumerate(T):
    plt.subplot(3,3,pos+1)
    plt.xlim((-3,3))
    plt.ylim((-3,3))
    plt.margins(0,0)
    plt.axis('off')
    i.tscales = [.5]*i.size
    i.drawNodes()
    i.drawLines()
    i.drawText()