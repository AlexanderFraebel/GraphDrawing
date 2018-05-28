from Graphs import *
from RandomPoints import randomNodes



def stronglyConnected(G):

    index = [np.Inf]*G.size
    lowlink = [np.Inf]*G.size
    S = []
    d = edgeDict(G.Mat)
    comps = []
    ctr = [0]

    for i in range(G.size):
        if index[i] == np.Inf:
            DFS(i,ctr,index,lowlink,S,comps,d)

    
    return comps

def DFS(v,ctr,index,lowlink,S,comps,d):

    index[v] = ctr[0]
    lowlink[v] = ctr[0]
    ctr[0] += 1
    S.append(v)

    
    for w in d[str(v)]:
        if index[w] == np.Inf:
            DFS(w,ctr,index,lowlink,S,comps,d)
            lowlink[v] = min(lowlink[v],lowlink[w])
        else:
            if w in S:
                lowlink[v] = min(lowlink[v],index[w])



    if lowlink[v] == index[v]:
       # print(v)
        #print(S)
        #print([index[i] for i in S])
        #print([lowlink[i] for i in S])
        L = []
        x = np.NaN
        while x != v:
            x = S.pop()
            L.append(x)
        comps.append(L)
       # print(comps)
        #print()
    
def testStronglyConnected():
    
    G = randomNodes(10,.5,NodeSize=.2,TextSize=1.5,seed=44345)
    G.addEdges([8,3,6,4,1,1,7,9,7,2,2,9,2,4,7],
               [3,6,4,8,4,7,5,7,2,0,6,0,9,3,4],directed=True)
    
    
    makeCanvas()
    G.drawNodes()
    G.drawText()
    G.drawArrows()
    
    
    S = stronglyConnected(G)
    
    color = ['pink','cornflowerblue','khaki','lightgreen','violet']
    ctr = 0
    for i in S:
        for j in i:
            G.colors[j] = color[ctr]
        ctr += 1
    
    G.drawNodes()