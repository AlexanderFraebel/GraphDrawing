from Graphs import *
import numpy as np

def petersonGraph(**kwargs):

    G = Graph(**kwargs)
    
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=2,n=5):
        G.addNode(i)
    
    G.addEdgesBi([0,1,2,3,4],[1,2,3,4,0])
    
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=1,n=5):
        G.addNode(i)
        G.addEdgesBi([ctr],[ctr-5])
        ctr += 1
    
    G.addEdgesBi([5,6,7,8,9],[7,8,9,5,6])
    
    return G


def bullGraph(**kwargs):
    
    G = Graph(**kwargs)
    
    ps = [[0,-1],[-.7,0],[.7,0],[-2,.5],[2,.5]]
    
    for i in range(len(ps)):
        G.addNode(ps[i])
        
    G.addEdgesBi([0,0,1,1,2],[1,2,2,3,4])
    
    return G

def bowtieGraph(**kwargs):
    
    G = Graph(**kwargs)
    
    ps = [[0,0],[-1,.5],[-1,-.5],[1,.5],[1,-.5]]
    
    for i in range(len(ps)):
        G.addNode(ps[i])
        
    G.addEdgesBi([0,0,0,0,1,3],[1,2,3,4,2,4])
    
    return G

def flowerSnarkGraph(**kwargs):
    
    G = Graph(**kwargs)
    
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=.7,n=5):
        G.addNode(i)
    
    G.addEdgesBi([0,1,2,3,4],[1,2,3,4,0])
        
    for i in arc([0,0],th=[np.pi/10,np.pi*2+(np.pi/10)],r=2.2,n=15):
        G.addNode(i)
    
    G.addEdgesBi([i for i in range(5,19)],[i for i in range(6,20)])
    G.addEdgesBi([5],[19])
    G.addEdgesBi([0,1,2,3,4],[5,8,11,14,17])
    G.addEdgesBi([6,7,9,10,13],[16,12,19,15,18])
    
    return G

def hypercubeGraph(**kwargs):
    G = Graph(**kwargs)
    
    for i in arc([0,0],th=[0,np.pi*2],r=1.2,n=8):
        G.addNode(i)
    
    for i in arc([0,0],th=[0,np.pi*2],r=2.5,n=8):
        G.addNode(i)
    
    G.addEdgesBi([0,0,1,1,2,2,3,4],[3,5,4,6,5,7,6,7])
    G.addEdgesBi([0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7],
                 [9,15,8,10,9,11,10,12,11,13,12,14,13,15,8,14])
    G.addEdgesBi([i for i in range(8,16)],[i for i in range(9,16)]+[8])
    return G

def completeGraph(N,**kwargs):
    
    xy = arc((0,0),2.5,[0,np.pi*2],N)
    G = Graph(**kwargs)
    for i,pos in enumerate(xy):
        G.addNode(pos)
    
    for i in range(N):
        G.addEdgesBi([i]*i,[x for x in range(i)])
        
    return G

def completeBipartiteGraph(P,Q,**kwargs):
    G = Graph(**kwargs)
        
    pp = np.linspace(-2.5,2.5,P)
    qq = np.linspace(-2.5,2.5,Q)
    
    for i in pp:
        G.addNode([i,1])
        
    for i in qq:
        G.addNode([i,-1])
        G.addEdges([G.size-1]*P,[i for i in range(P)])
    
    
    return G

def starGraph(N,**kwargs):
    G = Graph(**kwargs)
    
    G.addNode()
        
    xy = arc((0,0),2.5,[0,np.pi*2],N)

    for pos in xy:
        G.addNode(pos)
    G.addEdgesBi(0,[i+1 for i in range(N)])
        
    
    return G

def cycleGraph(N,**kwargs):
    G = Graph(**kwargs)
        
    xy = arc((0,0),2.5,[0,np.pi*2],N)

    for pos in xy:
        G.addNode(pos)
    G.addEdgesBi([i for i in range(N)],[i+1 for i in range(N-1)]+[0])
        
    
    return G

def pappusGraph(**kwargs):
    G = Graph(**kwargs)
    
    xy1 = arc((0,0),.6,[0,np.pi*2],6)
    xy2 = arc((0,0),1.8,[0,np.pi*2],6)
    xy3 = arc((0,0),2.4,[0,np.pi*2],6)
    
    for pos in xy1:
        G.addNode(pos)
        
    for pos in xy2:
        G.addNode(pos)
        
    for pos in xy3:
        G.addNode(pos)

    
    A = [0,1,2,6,7,8,9,10,11,12,13,14,15,16,17]
    B = [3,4,5,12,13,14,15,16,17,13,14,15,16,17,12]
    G.addEdges(A,B)
    
    A = [0,0,1,1,2,2,3,3,4,4,5,5]
    B = [7,11,6,8,7,9,8,10,9,11,6,10]
    G.addEdges(A,B)
    

    return G
        
    
###############################################################################
###############################################################################
##
## TESTING
##
###############################################################################
###############################################################################

def testParticularGraphs():

    G = pappusGraph(NodeSize=.15,TextSize=1.5)
    f,a = makeCanvas(size=[9,9])
    G.drawNodes()
    G.drawLines()
    G.drawText()
    #plt.title("Original Graph")
    print()

    #makeCanvas()
    #G.Mat = complement(G.Mat)
    #G.QuickDraw()
    #plt.title("Complementary Graph")

testParticularGraphs()