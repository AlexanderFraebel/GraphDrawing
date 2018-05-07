from Graphs import *
import numpy as np
import random


def isclose(P,L,d):
    for i in L:
        if dist(P,i) < d:
            return True
    return False


G = Graph(NodeSize=.2)

N = 20
sc = 50
xy = []
pos = np.linspace(-2.5,2.5,sc)

for i in range(N):
    ctr = 0
    xt = np.random.choice(pos)
    yt = np.random.choice(pos)
    while isclose([xt,yt],G.pos,.8):
        xt = np.random.choice(pos)
        yt = np.random.choice(pos)
        ctr += 1
        if ctr > 100:
            break

    print(ctr)
    G.addNode([xt,yt])  
    
    
makeCanvas()
G.drawNodes()

#for i in range(N):
#    for j in range(i):
#        print(dist(G.pos[i],G.pos[j]))
