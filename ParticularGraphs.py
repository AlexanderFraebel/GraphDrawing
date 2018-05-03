from Graphs import *
import numpy as np

def petersonGraph():

    G = Graph(rdef=.2)
    
    ctr = 0
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=2,n=5):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    G.addEdgesBi([0,1,2,3,4],[1,2,3,4,0])
    
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=1,n=5):
        G.addNode(i,text=str(ctr))
        G.addEdgesBi([ctr],[ctr-5])
        ctr += 1
    
    G.addEdgesBi([5,6,7,8,9],[7,8,9,5,6])
    
    return G


def bullGraph():
    
    G = Graph(rdef=.2)
    
    ps = [[0,-1],[-.7,0],[.7,0],[-2,.5],[2,.5]]
    
    for i in range(len(ps)):
        G.addNode(ps[i],text=str(i))
        
    G.addEdgesBi([0,0,1,1,2],[1,2,2,3,4])
    
    return G

def bowtieGraph():
    
    G = Graph(rdef=.2)
    
    ps = [[0,0],[-1,.5],[-1,-.5],[1,.5],[1,-.5]]
    
    for i in range(len(ps)):
        G.addNode(ps[i],text=str(i))
        
    G.addEdgesBi([0,0,0,0,1,3],[1,2,3,4,2,4])
    
    return G

def flowerSnarkGraph():
    
    G = Graph(rdef=.2)
    
    ctr = 0
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=.7,n=5):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    G.addEdgesBi([0,1,2,3,4],[1,2,3,4,0])
        
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=2.2,n=15):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    G.addEdgesBi([i for i in range(5,19)],[i for i in range(6,20)])
    G.addEdgesBi([5],[19])
    G.addEdgesBi([0,1,2,3,4],[5,8,11,14,17])
    G.addEdgesBi([6,7,9,10,13],[16,12,19,15,18])
    
    return G


###############################################################################
###############################################################################
##
## TESTING
##
###############################################################################
###############################################################################


fig1, ax1 = makeCanvas()
#G = bullGraph()
#G = petersonGraph()
#G = bowtieGraph()
G = flowerSnarkGraph()
G.QuickDraw(fig1,ax1)