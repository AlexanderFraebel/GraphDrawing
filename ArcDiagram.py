from Graphs import *
import numpy as np

def arcDiagram(R,L=[None],title="",size=[7,7],nodeSize=.3,
                      nodeCol = (.53, .81, .94), lineSize  = 2, lineCol = "black"):
    
    fig, ax = makeCanvas(xlim=[-3,3],ylim=[-3,3],size=size)
    
    n = R.shape[0]
    if len(L) != n:
        L = [str(i) for i in range(n)]

    G = Graph(rdef=nodeSize,tscaledef=15,coldef = nodeCol)
    
    pos = np.linspace(-2.5,2.5,n)
    
    for p,t in zip(pos,L):
        G.addNode([p,0],text=t)
    
    #print(R)
    for i in np.argwhere(R != 0):
        A,B = i[0],i[1]
        if A == B:
            continue
        #print(A,B)
        d = abs(pos[A]-pos[B])/2
        m = (pos[A]+pos[B])/2
        a = arcXY([m,nodeSize/3],r=d)
        plt.plot(a[0],a[1],color=lineCol,zorder=0,lw=lineSize)
        
    G.drawNodes(fig,ax)
    return G, fig, ax
    
R = randAjdMat(7)
arcDiagram(R)