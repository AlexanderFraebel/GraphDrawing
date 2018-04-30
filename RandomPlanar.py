from Graphs import *
import numpy as np
import random

N = 5

sc = 20
X = []
Y = []
M = np.ones([sc,sc])
pos = np.linspace(-2.5,2.5,sc)
for i in range(N):
    
    ar = np.argwhere(M==1)
    c = np.random.choice(len(ar))
    
    p1,p2 = ar[c][0],ar[c][1]
    
    X.append(pos[p1])
    Y.append(pos[p2])
    
    print(p1,p2)
    M[p1] = 0
    M[:,p2] = 0
    M[p2] = 0
    M[:,p1] = 0
    #print(M)
    
#print(X,Y)

G = Graph(rdef=.2,xlims=[-3,3],ylims=[-3,3])

for xy in zip(X,Y):
    #print(xy)
    G.addNode(xy)
    
G.drawNodes()

def lineeq(A,B):
    m = (A.y-B.y)/(A.x-B.x)
    b = A.y-m*A.x
    xlim = [min(A.x,B.x),max(A.x,B.x)]
    return m,b,xlim

def lineinter(m1,b1,xlim1,m2,b2,xlim2):
    x = (b2-b1)/(m1-m2)
    if x > xlim1[0] and x < xlim1[1] and x > xlim2[0] and x < xlim2[1]:
        return True
    return False

lines = []
r = [i for i in range(5)]
for i in range(5):
    a,b = np.random.choice(r,2,replace=False)
    connect(G.Nodes[a],G.Nodes[b])
    
    m,inter,x = lineeq(G.Nodes[a],G.Nodes[b])
    #print(m,inter,x)
    print(inter)
    
    if i > 1:
        for L in lines:
            if lineinter(m,inter,x,L[0],L[1],L[2]):
                connect(G.Nodes[a],G.Nodes[b],col='red',width=2)
    
    lines.append(lineeq(G.Nodes[a],G.Nodes[b]))
    