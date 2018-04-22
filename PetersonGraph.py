from Graphs import *

G = Graph(rdef=.3)

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

G.QuickDraw()
