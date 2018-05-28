from Graphs import *
from RandomPoints import randomNodes
from StronglyConnected import stronglyConnected
from matplotlib.pyplot import cm
from BiconnectedComponents import biconnected

def unitDiskGraph(N,f,d,xlim = [-2.8,2.8], ylim = [-2.8,2.8], seed = None,
                **kwargs):
    G = randomNodes(N,f,xlim=xlim,ylim=ylim,seed=seed,**kwargs)
    
    for i in range(N):
        for j in range(i):
            if dist(G.pos[i],G.pos[j]) < d:
                G.addEdges(i,j)
    
    return G

def testUnitDiskGraph(seed=0):

    makeCanvas(size=[8,8])
    un = 1
    G = unitDiskGraph(40,.5,un,seed=seed,NodeSize=.15)
    G.drawLines()
    G.drawNodes()
    
    ax = plt.gca()
    for i in G.pos:
        circ = plt.Circle(i,radius = un/2, fc = '#00000000', ec = 'lightgray', zorder=0)
        ax.add_patch(circ)
    plt.title("Unit Disk Graph as Circle Intersections")
        
    
    makeCanvas(size=[8,8])
    un = 1
    G = unitDiskGraph(40,.5,un,seed=seed,NodeSize=.15)
    
    for i in range(G.size):
        for j in range(i):
            if G.Mat[j,i] == 1 and G.Mat[i,j] == 1:
                G.Mat[j,i] = random.choice([0,0,1])
    
    G.drawArrows(hwd=.05,hln=.05)
    
    S = stronglyConnected(G)
    
    ctr = 0
    C = cm.tab20([i for i in range(20)])
    for i in S:
        for j in i:
            G.colors[j] = C[ctr%20]
            G.texts[j] = ctr
        ctr += 1
    G.drawNodes()
    G.drawText()
    plt.title("Strongly Connected Components\n(After Randomizing Directionality)")
    

       
testUnitDiskGraph(35443)