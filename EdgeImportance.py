from Graphs import *
from DijkstrasAlgorithm import dijkstra
from RandomPoints import randomNodes

G = randomNodes(14,.5,NodeSize=.2,seed=44345)
G.addEdges([8,3,6,4,1,1,7,9,7,2,2,9,2, 5,10, 0,11, 4,12, 1],
           [3,6,4,8,4,7,5,7,2,0,6,0,9,10, 0,11, 3,12, 8,13],directed=False)

makeCanvas()
G.drawNodes()
G.drawText()

D = maskDist(G)

M = np.zeros([G.size,G.size])

for n in range(10):
    x = dijkstra(D,n)
    x = x[1]
    used = []
    for i in x:
        for j in range(len(i)-1):
            if [i[j],i[j+1]] not in used:
                used.append([i[j],i[j+1]])

    for x in used:
        M[x[0],x[1]] += 1
        M[x[1],x[0]] += 1
        
print(M)

for i in np.argwhere(M > 0):
    if i[0] == i[1]:
        continue
    connect(G.pos[i[0]],G.pos[i[1]],col='black',width=M[i[0],i[1]]/1.5)