from Graphs import *
import numpy as np


def isclose(P,L,d):
    for i in L:
        if dist(P,i) < d:
            return True
    return False


def randomNodes(N,d,xlim = [-2.8,2.8], ylim = [-2.8,2.8],
                NodeSize=.2, TextSize=1, NodeColor=(.53, .81, .94), seed=None):
    if seed != None:
        np.random.seed(seed)
    G = Graph(NodeSize=NodeSize,TextSize=TextSize,NodeColor=NodeColor)
    
    for i in range(N):
        ctr = 0
        x = np.random.uniform(xlim[0],xlim[1],1)
        y = np.random.uniform(ylim[0],ylim[1],1)
        while isclose([x,y],G.pos,d):
            x = np.random.uniform(xlim[0],xlim[1],1)
            y = np.random.uniform(ylim[0],ylim[1],1)
            ctr += 1
            if ctr > 200:
                break
    
        G.addNode([x[0],y[0]])  
    
    return G

def testRandomPoints():
    makeCanvas()
    
    G = randomNodes(50,.3)
    G.drawNodes()
    
#testRandomPoints()