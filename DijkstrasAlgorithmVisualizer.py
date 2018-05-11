from Graphs import *
from DijkstrasAlgorithm import dijkstra
from ParticularGraphs import flowerSnarkGraph
from RandomPoints import randomNodes

def DijkstraVisualizer(G,N=0,text=True):
    
    makeCanvas()
    distances,paths = dijkstra(G.Mat,N)
    
    G.colors[N] ='orange'
    G.drawArrows(col='lightgray')
    G.drawText()
    G.drawNodes()
    if text == True:
        print("\nPaths from Node {}".format(N))
        for pos,val in enumerate(paths):
            s = ""
            if len(val) == 0:
                print("{}: Not Connected".format(pos))
                continue
            for i in val:
                s += str(i) + " \u2192 "
            print("{}: {}".format(pos,s[:-2]))
        
        print("\nDistances from Node {}".format(N))
        for pos,val in enumerate(distances):
            print("{:<2}: {:.3f}".format(pos,val),end="   ")
            if (pos+1) % 3 == 0:
                print("")
        print()
    
    for i in paths:
        if len(i) == 1:
            continue
        for j in range(len(i)-1):
            connectArr(G.pos[i[j]],G.pos[i[j+1]],headpos=G.radii[0],col='red',z=0,width=2)
    

    return G


def testDijkstrasAlgorithmVisualizer():
    G = flowerSnarkGraph()
    G = DistanceMatrix(G)
    DijkstraVisualizer(G,7,text=False)
    
    

    N = 15
    G = randomNodes(N,1,NodeSize=.15,TextSize=1.5)
    G.Mat = connectedGraph(N,prob=.1)
    DijkstraVisualizer(G,text=False)

testDijkstrasAlgorithmVisualizer()