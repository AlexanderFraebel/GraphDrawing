from Graphs import *
import random
from ParticularGraphs import hypercubeGraph, flowerSnarkGraph

# Performs a depth first search where the adjacent node with the lowest index
# is always chosen next.
def depthTree(R,x):
    checked = []
    working = [x]
    d = edgeDict(R)
    cur = x
    preds = [x]*len(d)
    while len(working) > 0:
        cur = working[-1]
        new = False
        for i in d[str(cur)]:
            if i not in checked and i not in working:
                working.append(i)
                new = True
                preds[i] = cur
                break
        if new == False:
            checked.append(working.pop())

    return preds

# Performs a breadth first search where the adjacent node with the lowest index
# is always chosen next.
def widthTree(R,x):
    checked = set()
    working = [x]
    d = edgeDict(R)
    preds = [x]*len(d)
    cur = x
    while len(working) > 0:
        cur = working.pop(0)
        for i in d[str(cur)]:
            if i not in checked and i not in working:
                working.append(i)
                preds[i] = cur
        checked.add(cur)
    return preds

def randomTree(R,x):
    checked = []
    working = [x]
    d = edgeDict(R)
    cur = x
    preds = [x]*len(d)
    while len(working) > 0:
        random.shuffle(working)
        cur = working[-1]
        new = False
        for i in d[str(cur)]:
            if i not in checked and i not in working:
                working.append(i)
                new = True
                preds[i] = cur
                break
        if new == False:
            checked.append(working.pop())

    return preds



def testTree():
    G = flowerSnarkGraph(NodeSize=.2)
    N = 8
    d1 = depthTree(G.Mat,N)
    d2 = widthTree(G.Mat,N)
    d3 = randomTree(G.Mat,N)
    G.colors[N] = 'orange'
    
    makeCanvas()
    
    G.QuickDraw()
    G.drawLines(col='lightgray')
    G.drawText()
    for i,j in enumerate(d1):
        if i == j:
            continue
        connectArr(G.pos[j],G.pos[i],headpos=.2,headwidth=.1,headlength=.1,col='black',width=3)
    plt.title("Depth First Search",size=30)


    makeCanvas()
    
    G.QuickDraw()
    G.drawLines(col='lightgray')
    G.drawText()
    for i,j in enumerate(d2):
        if i == j:
            continue
        connectArr(G.pos[j],G.pos[i],headpos=.2,headwidth=.1,headlength=.1,col='black',width=3)
    plt.title("Breadth First Search",size=30)
    
    
    
    makeCanvas()
    
    G.QuickDraw()
    G.drawLines(col='lightgray')
    G.drawText()
    for i,j in enumerate(d3):
        if i == j:
            continue
        connectArr(G.pos[j],G.pos[i],headpos=.2,headwidth=.1,headlength=.1,col='black',width=3)
    plt.title("Random Search",size=30)
    
#testTree()