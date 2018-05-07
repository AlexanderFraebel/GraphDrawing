from Graphs import *
import numpy as np

def petersonGraph():

    G = Graph(NodeSize=.2)
    
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
    
    G = Graph(NodeSize=.2)
    
    ps = [[0,-1],[-.7,0],[.7,0],[-2,.5],[2,.5]]
    
    for i in range(len(ps)):
        G.addNode(ps[i],text=str(i))
        
    G.addEdgesBi([0,0,1,1,2],[1,2,2,3,4])
    
    return G

def bowtieGraph():
    
    G = Graph(NodeSize=.2)
    
    ps = [[0,0],[-1,.5],[-1,-.5],[1,.5],[1,-.5]]
    
    for i in range(len(ps)):
        G.addNode(ps[i],text=str(i))
        
    G.addEdgesBi([0,0,0,0,1,3],[1,2,3,4,2,4])
    
    return G

def flowerSnarkGraph():
    
    G = Graph(NodeSize=.2)
    
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

def hypercubeGraph():
    G = Graph(NodeSize=.2)
    ctr = 0
    for i in arc([0,0],th=[0,np.pi*2],r=1.2,n=8):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    for i in arc([0,0],th=[0,np.pi*2],r=2.5,n=8):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    G.addEdgesBi([0,0,1,1,2,2,3,4],[3,5,4,6,5,7,6,7])
    G.addEdgesBi([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7],
                 [9,15,8,10,9,11,10,12,11,13,12,14,13,15,8,14])
    G.addEdgesBi([i for i in range(8,16)],[i for i in range(9,16)]+[8])
    return G

def completeGraph(N):
    
    xy = arc((0,0),2.5,[0,np.pi*2],N)
    G = Graph()
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str(i))
    
    for i in range(N):
        G.addEdgesBi([i]*i,[x for x in range(i)])
    return G

def completeBipartiteGraph(P,Q):
    G = Graph()
        
    pp = np.linspace(-2.5,2.5,P)
    qq = np.linspace(-2.5,2.5,Q)
    
    ctr = 0
    for i in pp:
        G.addNode([i,1],text=str(ctr))
        ctr += 1
        
    for i in qq:
        G.addNode([i,-1],text=str(ctr))
        G.addEdgesBi([ctr]*P,[i for i in range(P)])
        ctr += 1
    
    
    return G

###############################################################################
###############################################################################
##
## TESTING
##
###############################################################################
###############################################################################

def testfunc():
    #G = bullGraph()
    #G = petersonGraph()
    #G = bowtieGraph()
    #G = flowerSnarkGraph()
    #G = completeGraph(8)
    #G = completeBipartiteGraph(3,4)
    
    
    G = petersonGraph()
    for i in G.radii:
        i = .1
    print(G.radii)
    fig1, ax1 = makeCanvas()
    G.QuickDraw(fig1,ax1)
    plt.title("Original Graph")
    
    fig2, ax2 = makeCanvas()
    G.Mat = complement(G.Mat)
    G.QuickDraw(fig2,ax2)
    plt.title("Complementary Graph")

#testfunc()