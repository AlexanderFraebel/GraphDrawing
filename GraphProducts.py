from Graphs import *

G1 = Graph(NodeSize=.2)
G1.addNode([-2,0])
G1.addNode([-2,1.3])
G1.addNode([-2.7,-1])
G1.addNode([-2.2,-2.3])
G1.addEdges([0,0,0,2],[1,2,3,3])

G2 = Graph(NodeSize=.2)
G2.addNode([-1,0])
G2.addNode([0,0])
G2.addNode([1,.5])
G2.addNode([2,-.5])
G2.addNode([-1.5,.5])
G2.addEdges([0,1,1,0],[1,2,3,4])


def addGraphRoot(GA,GB,r1,r2):

    r = [i.copy() for i in GB.pos]
    x = GA.pos[r1][0]-GB.pos[r2][0]
    y = GA.pos[r1][1]-GB.pos[r2][1]
    
    for z in range(len(r)):
        r[z][0] += x
        r[z][1] += y
    del r[r2]
    
    G = GB.Mat.copy()
    rt = G[:,r2]
    G = np.delete(G,r2,0)
    G = np.delete(G,r2,1)
    
    GA.addNodes(r)

    GA.Mat[GA.size-GB.size+1:GA.size,GA.size-GB.size+1:GA.size] = G
    
    for pos,ed in enumerate(rt):
        if ed == 1:
            GA.addEdges([r1],[GA.size-GB.size+pos])

    
for i in range(G1.size):
    addGraphRoot(G1,G2,i,0)

print(edgeDict(G1.Mat))


makeCanvas(xlim=[-4,4],ylim=[-4,4],size=[10,10])
G1.drawNodes()
G1.drawLines()
G1.drawText()