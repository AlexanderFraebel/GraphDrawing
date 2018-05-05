from Graphs import *

def disconnected():
    R = np.zeros([18,18])
    for i in range(3):
        t = regularGraph(6,2)
        R[i*6:i*6+6,i*6:i*6+6] = t
    s = [i for i in range(18)]
    random.shuffle(s)
    p = np.argsort(s)
    R = R[:,p]
    R = R[p,:]
    return R


R = disconnected()
com = connectedComponents(R)
print(com)
G, fig, ax = connectogramUndir(R,lineSize=1,nodeSize=.2,curve=.5)

cls = ['red','orange','yellow','green','blue','purple']
ctr = 0
for x in com:
    for i in x:
        G.colors[i] = cls[ctr]
    ctr += 1
G.drawNodes(fig,ax)