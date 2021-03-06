from Graphs import *

## Lowpoint of vertex "v" is the lowest depth of all descendats of "v" and of
## the neighbors of "v" other than its parent (in the dfs ordering)

def biconnected(G):
    x = 0
    
    # Create a dictionary of edges
    d = edgeDict(G.Mat)
    
    # The traits we need to keep track of.
    pred = [np.NAN]*len(d)
    depth = [np.NAN]*len(d)
    lowpoint = [np.NAN]*len(d)
    cut = [False]*len(d)
    
    # Will store and output the connected edges and the vertices that define
    # the components
    edges = []
    verts = []
    
    # For keeping track of where we are and have been in the graph
    checked = []
    working = [0]
    
    # The explore function is recursive
    for i in range(G.size):
        t = [sublist for sublist in verts]
        if i in t:
            continue
        working = [i]
        explore(d,i,checked,working,pred,depth,lowpoint,cut,edges,verts)
    
        # Check if there is anything left
        v = set()
        if len(edges) > 0:
            #print("Component: ",end="")
            while len(edges) > 0:
                w = edges.pop()
                v.add(w[0])
                v.add(w[1])
                #print(w,end="")
        if len(v) > 0:
            verts.append(v)
    
    #print("\n\nBiconnected Subgraphs:\n{}".format(verts))
    
    #print("\n")
    #nodes = [i for i in range(len(d))]
    #for i in zip(nodes,pred,depth,lowpoint,cut):
    #    print(i)

    
    return verts

def explore(d,x,checked,working,pred,depth,lowpoint,cut,edges,verts):

    # Counting the number of children, we only really care about this for the
    # starting point.
    ch = 0
    lowpoint[x] = len(working)-1
    depth[x] = len(working)-1
    
    if len(working) > 0:
        for i in d[str(x)]:
            if i not in checked and i not in working:
                # We will now visit this node so store it in working
                working.append(i)
                # The parent of this node is x
                pred[i] = x
                # It is a child of x
                ch += 1
                # Store edge between them
                edges.append((x,i))
                
                # Now continue to explore the graph
                explore(d,working[-1],checked,working,pred,depth,lowpoint,cut,edges,verts)

                # Having finished exploring below it update the lowest point
                # seen by the children of x
                lowpoint[x] = min(lowpoint[i],lowpoint[x])
                
                if depth[x] == 0 and ch > 1 or depth[x] > 0 and lowpoint[i] >= depth[x]:
                    cut[x] = True
                    #print("Component: ".format(x),end="")
                    w = 0
                    v = set()
                    while w != (x,i):
                        w = edges.pop()
                        v.add(w[0])
                        v.add(w[1])
                        #print(w,end="")
                    verts.append(v)
                   # print()
                
                
            # If the node has been visited before
            elif i != pred[x] and lowpoint[x] > depth[i]:
                lowpoint[x] = min(lowpoint[x],depth[i])
                edges.append((x,i))

    # Move node from working to checked, we're done with it
    checked.append(working.pop())
    


        

###############################################################################
###############################################################################
def testBiconnectedComponents():
    G = Graph(NodeSize=.2)
    
    pos = [[-3,2],[-3,1],[-2,1],[-2,0],[-1,-1],
           [-1,0],[0,1],[0,0],[1,0],[-1,2],[0,2],
           [0,3],[1,2],[1,-1],[2,1]]
    
    for i,p in enumerate(pos):
        G.addNode(p,text=i)
    
    
    G.addEdgesBi(
            [0,0,1,2,3,4,5,6,6,6,9,9,11,10,12,13,14,14],
            [1,2,3,3,4,5,6,7,8,9,10,11,10,12,8,8,12,8])
    
    
    makeCanvas(xlim=[-4,4],ylim=[-4,4],size=[10,10])
    G.QuickDraw()
    
    
    #d = dfs(G.Mat,5)
    #for i,j in d:
        #p = G.Nodes[i].xy
        #plt.text(p[0]+.05,p[1]+.3,str(j),color='red',size="x-large")
    
        
    v = biconnected(G)
    print(v)

    for i in v:
        convexCircle(G,list(i),.3,'red')
        
testBiconnectedComponents()
    