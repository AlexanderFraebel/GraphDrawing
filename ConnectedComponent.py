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

# Explore the graph staring at vertex x using a width first search
def explore(R,x):
    checked = set()
    working = [x]
    d = edgeDict(R)
    while len(working) > 0:
        cur = working.pop(0)
        for i in d[str(cur)]:
            if i not in checked:
                working.append(i)
        checked.add(cur)
    return sorted(list(checked))

# For each element check if it is part of a known connected component. If it is
# check the next one. If it isn't then explore the graph from that vertex and
# return the verticies of the connected component it is a part of.
def connectedComponents(R):
    n = np.shape(R)[0]
    out, L = [],[]
    ctr = 0
    while len(L) != n:
        if ctr in L:
            ctr += 1
            continue
        else:
            T = explore(R,ctr)
            out.append(T)
            L += T
    return out

#R = regularGraph(17,2)
R = disconnected()
connectogramUndir(R,lineSize=1,nodeSize=.2,curve=.5)
print(connectedComponents(R))

