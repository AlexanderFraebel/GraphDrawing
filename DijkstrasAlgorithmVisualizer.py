from Graphs import *
from DijkstrasAlgorithm import dijkstra
from ParticularGraphs import flowerSnarkGraph

def DijkstraVisualizer(G,N=0):
    
    fig, ax = makeCanvas()
    di = dijkstra(G.Mat,N)
    
    G.colors[N] ='orange'
    
    for pos,val in enumerate(di[1]):
        s = ""
        if len(val) == 0:
            print("{}: Not Connected".format(pos))
            continue
        for i in val:
            s += str(i) + " \u2192 "
        print("{}: {}".format(pos,s[:-2]))
        
    for i in di[1]:
        if len(i) == 1:
            continue
        for j in range(len(i)-1):
            connectArr(G.pos[i[j]],G.pos[i[j+1]],headpos=G.radii[0],col='red',z=0,width=2)
    
    G.QuickDraw(fig,ax)

DijkstraVisualizer(flowerSnarkGraph(),7)