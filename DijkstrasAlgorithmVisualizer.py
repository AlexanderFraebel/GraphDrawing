from Graphs import *
from DijkstrasAlgorithm import dijkstra
from ParticularGraphs import flowerSnarkGraph

def DijkstraVisualizer(G,N=0):
    
    fig, ax = makeCanvas()
    distances,paths = dijkstra(G.Mat,N)
    
    G.colors[N] ='orange'
    
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
    
    G.QuickDraw()
    G.drawText()
    return G

G = flowerSnarkGraph()
G = DistanceMatrix(G)

DijkstraVisualizer(G,7)