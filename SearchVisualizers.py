from Graphs import *
from PlatonicSolids import dodecahedronGraph
from ParticularGraphs import petersonGraph

## DO NOT USE IM PRETTY SURE THIS IS BROKEN ##

def DFSVisualizer(G,N=0):
    
    fig, ax = makeCanvas()
    dep = dfs(G.Mat,N)
    
    G.Nodes[N].update(col='orange')
    
    G.QuickDraw(fig,ax)
    
    print(dep)
    
    for i in range(len(dep)):
        d = dep[i]
        for j in range(len(dep)):
            if dep[j] == d-1:
                connectArr(G.Nodes[j],G.Nodes[i],col='red',z=0,width=2,headwidth=.1,headlength=.1)
    


DFSVisualizer(dodecahedronGraph())
DFSVisualizer(petersonGraph())
