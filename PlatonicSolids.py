###############################################################################
###############################################################################
##
## NETS OF PLATONIC SOLIDS
##
###############################################################################
###############################################################################
    
def tetrahedronGraph():
    
    G = Graph(rdef=.2)
    
    ps = [[0,-.3],[-1,-1],[1,-1],[0,.7]]
    
    for i in range(len(ps)):
        G.addNode(ps[i],text=str(i))
    
    G.addEdgesBi([0,0,0,1,2,3],[1,2,3,2,3,1])
    
    return G

def cubeGraph():
    
    G = Graph(rdef=.2)
    
    ps = [[-1,-1],[-1,1],[1,1],[1,-1],[-2,-2],[-2,2],[2,2],[2,-2]]
    
    for i in range(len(ps)):
        G.addNode(ps[i],text=str(i))
    
    G.addEdgesBi([0,1,2,3,4,5,6,7,0,1,2,3],[1,2,3,0,5,6,7,4,4,5,6,7])
    
    return G

def octahedronGraph():
    
    G = Graph(rdef=.2)
    
    ps = [[-.6,-.4],[.6,-.4],[0,.7],[-2,1.5],[2,1.5],[0,-2.5]]
    
    for i in range(len(ps)):
        G.addNode(ps[i],text=str(i))
    
    G.addEdgesBi([0,1,2,3,4,5,3,3,4,4,5,5],[1,2,0,4,5,3,2,0,2,1,0,1])
    
    return G

def dodecahedronGraph():
    
    G = Graph(rdef=.2)
    
    rt = -np.pi/2
    ctr = 0
    for i in arc([0,0],th=[rt,np.pi*2+(rt)],r=.6,n=5):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    rt = -np.pi/2
    for i in arc([0,0],th=[rt,np.pi*2+(rt)],r=1.2,n=5):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    rt = np.pi/2
    for i in arc([0,0],th=[rt,np.pi*2+(rt)],r=1.9,n=5):
        G.addNode(i,text=str(ctr))
        ctr += 1
        
    rt = np.pi/2
    for i in arc([0,0],th=[rt,np.pi*2+(rt)],r=2.6,n=5):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    G.addEdgesBi([0,1,2,3,4],[1,2,3,4,0])
    G.addEdgesBi([0,1,2,3,4],[5,6,7,8,9])
    G.addEdgesBi([10,10,11,11,12,12,13,13,14,14],[7,8,8,9,9,5,5,6,6,7])
    G.addEdgesBi([10,11,12,13,14],[15,16,17,18,19])
    G.addEdgesBi([15,16,17,18,19],[16,17,18,19,15])
    
    return G

def icosahedronGraph():
    
    G = Graph(rdef=.2)
    
    rt = -np.pi/2
    ctr = 0
    for i in arc([0,0],th=[rt,np.pi*2+(rt)],r=.3,n=3):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    rt = -np.pi/2
    for i in arc([0,0],th=[rt,np.pi*2+(rt)],r=.9,n=6):
        G.addNode(i,text=str(ctr))
        ctr += 1
    
    rt = np.pi/2
    for i in arc([0,0],th=[rt,np.pi*2+(rt)],r=2.6,n=3):
        G.addNode(i,text=str(ctr))
        ctr += 1
        
    G.addEdgesBi([0,1,2,3,4,5,6,7,8,9,10,11],[1,2,0,4,5,6,7,8,3,10,11,9])
    G.addEdgesBi([9,9,9,10,10,10,11,11,11],[5,6,7,3,7,8,3,4,5])
    G.addEdgesBi([0,0,0,1,1,1,2,2,2],[3,4,8,4,5,6,6,7,8])
    
    return G

###############################################################################
###############################################################################
##
## TESTING
##
###############################################################################
###############################################################################

fig1 = plt.figure(1)
fig1.set_size_inches(10,15)

ax = plt.subplot(321)
ax.axis('off')
G = tetrahedronGraph()
G.QuickDraw(fig1,ax)

ax = plt.subplot(322)
ax.axis('off')
G = cubeGraph()
G.QuickDraw(fig1,ax)

ax = plt.subplot(323)
ax.axis('off')
G = octahedronGraph()
G.QuickDraw(fig1,ax)

ax = plt.subplot(324)
ax.axis('off')
G = dodecahedronGraph()
G.QuickDraw(fig1,ax)

ax = plt.subplot(325)
ax.axis('off')
G = icosahedronGraph()
G.QuickDraw(fig1,ax)