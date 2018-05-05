import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as patches

class Graph:
    def __init__(self,rdef=.3,tscaledef=1,coldef=(.53, .81, .94)):
        
        # Set a few default characteristics
        self.rdef = rdef
        self.tscaledef = tscaledef
        self.coldef = coldef
        
        # Prepare a list of nodes
        self.size = 0
        
        # Need an easier system to control than each node being an independent
        # object
        self.colors = []
        self.pos = []
        self.radii = []
        self.texts = []
        self.tscales = []
        self.zpos = []
        
        # Prepare a matrix of connections
        self.Mat = np.asarray([0])
    
    ## Create new node
    def addNode(self,xy=[0,0],r=None,col=[None],text="",tscale=None,z=1):
        if r == None:
            r = self.rdef
        if tscale == None:
            tscale = self.tscaledef
        if any(i == None for i in col):
            col = self.coldef
            
        self.radii.append(r)
        self.colors.append(col)
        self.pos.append(xy)
        self.texts.append(text)
        self.tscales.append(tscale)
        self.zpos.append(z)
        
        self.size += 1
  
        # Expand the adjacency matrix
        t = np.zeros((self.size,self.size))
        t[:-1,:-1] = self.Mat
        self.Mat = t
        
    ## Create edges. By default just sets them equal to 1.
    def addEdges(self,A,B,D=[None]):
        if any(i == None for i in D):
            D = [1]*len(A)
        self.Mat[A,B] = D
    
    # Bidirectional edges. Often doesn't matter when drawing but I figured it 
    # would be annoying if this wasn't in here somewhere.
    def addEdgesBi(self,A,B,D=[None]):
        if any(i == None for i in D):
            D = [1]*len(A)
        self.Mat[A,B] = D
        self.Mat[B,A] = D
    
    ## Remove nodes and edges.
    def delNode(self,n):
        del self.colors[n]
        del self.pos[n]
        del self.radii[n]
        del self.texts[n]
        del self.tscales[n]
        del self.zpos[n]
        self.Mat = np.delete(self.Mat,n,0)
        self.Mat = np.delete(self.Mat,n,1)
    
    def delEdges(self,A,B):
        self.Mat[A,B] = 0
    
    def delEdgesBi(self,A,B):
        self.Mat[A,B] = 0
        self.Mat[B,A] = 0

    # Simple drawing functions for common situations
    def QuickDraw(self,fig,ax):
        self.drawNodes(fig,ax)
        self.drawLines()
        
    def drawNodes(self,fig,ax):
        for i in range(self.size):
            circ = plt.Circle(self.pos[i],radius = self.radii[i], 
                              fc = self.colors[i],zorder=self.zpos[i])
            ax.add_patch(circ)

    
    def drawText(self,fig,ax):
        s = fig.get_size_inches()
        d = np.sqrt(s[0]**2+s[1]**2)*5
        for i in range(self.size):
            plt.text(self.pos[i][0],self.pos[i][1],self.texts[i],
                     size=self.radii[i]*self.tscales[i]*d,
                     ha='center',va='center',zorder=self.zpos[i])
    
    def drawArrows(self,col="black",wd=2,hwd=.1,hln=.2,term=None):
        for i in np.argwhere(self.Mat != 0):
            if i[0] == i[1]:
                continue
            connectArr(self.pos[i[0]],self.pos[i[1]],headpos=self.radii[i[1]],
                          width=wd,headwidth=hwd,headlength=hln,z=0,col=col)
    
    def drawLines(self,col="black",wd=2):
        for i in np.argwhere(self.Mat != 0):
            if i[0] == i[1]:
                continue
            connect(self.pos[i[0]],self.pos[i[1]],col=col,width=wd)
    
    def drawCurve(self,r=1,col="black",wd=2):
        for i in np.argwhere(self.Mat != 0):
            if i[0] == i[1]:
                continue
            bezierCurve(self.pos[i[0]],self.pos[i[0]],r=1,col=col,lw=wd)

###############################################################################
###############################################################################
##
## CREATE PLOTS
##
###############################################################################
###############################################################################

def makeCanvas(xlim=[-3,3],ylim=[-3,3],size=[7,7]):
    fig = plt.figure()
    fig.set_size_inches(size[0], size[1])
    ax = plt.axes(xlim=xlim, ylim=ylim)
    ax.axis('off')
    return fig, ax

###############################################################################
###############################################################################
##
## DRAWING CONNECTING LINES
##
###############################################################################
###############################################################################
            
def connect(A,B,col="black",width=1,z=0):
    if type(A) == list and type(B) == list or type(A) == np.ndarray and type(B) == np.ndarray:
        if len(A) == 2 and len(B) == 2:
            plt.plot([A[0],B[0]],[A[1],B[1]],color=col,lw=width,zorder=z)
        else:
            raise ValueError("A and B must length 2")    
    else:
        raise ValueError("A and B do not match or are not recognized")

def connectArr(A,B,col="black",width=1,headpos=0,headwidth=.2,headlength=.2,z=0):

    if type(A) == list and type(B) == list or type(A) == np.ndarray and type(B) == np.ndarray:
        if len(A) == 2 and len(B) == 2:
            ter = distpt(A,B,headpos)
            plt.arrow(A[0],A[1],ter[0]-A[0],ter[1]-A[1],color=col,lw=width,zorder=z,
                      head_width=headwidth, head_length=headlength,
                      length_includes_head=True)
        else:
            raise ValueError("A and B must length 2")  
    else:
        raise ValueError("A and B do not match or are not recognized")


# Connection to self
# Creates a circle the intersects the center of the node
# th controls the position of the circle and rot controls the position of the
# arrow around the loop
def loop(A,r=.4,th=0,rot=0,col="black",width=1,headwidth=.2,headlength=.2,z=0):
    a = np.linspace(0,np.pi*2)
    x = r*np.cos(a+rot)+A[0]+(np.cos(th)*r)
    y = r*np.sin(a+rot)+A[1]+(np.sin(th)*r)
    plt.plot(x,y,color=col,lw=width,zorder=z)
    connectArr([x[0],y[0]],[x[1],y[1]],headwidth=.2,headlength=.2,z=z)


###############################################################################
###############################################################################
##
## CURVED LINES
##
###############################################################################
###############################################################################

# Define arbitrary Bezier splines with either one or two control points
def bezierQuad(A,B,C):
    t = np.linspace(0,1,50)
    P0 = perpt(A,B,t)
    P1 = perpt(B,C,t)
    return perpt(P0,P1,t)

def bezierCube(A,B,C,D):
    t = np.linspace(0,1,50)
    P0 = bezierQuad(A,B,C)
    P1 = bezierQuad(B,C,D)
    return perpt(P0,P1,t)
    
# Simplified Bezier spline. Control point is at a distance perpendicular from
# the midpoint of the connecting line.
def bezierCurve(A,B,r=1,color="black",lw=2,z=0):
    t = np.linspace(0,1,50)
    mdpt = midpt(A,B)
    if A[0] == B[0]:
        R = [A[0]+r,midY(A,B)]
    if A[1] == B[1]:
        R = [midX(A,B),A[1]+r]
    if A[0] != B[0] and A[1] != B[1]:
        ang = np.arctan2((A[1]-B[1]),(A[0]-B[0]))+np.pi/2
        R = [r*np.cos(ang)+mdpt[0],r*np.sin(ang)+mdpt[1]]
    P0 = perpt(A,R,t)
    P1 = perpt(R,B,t)
    
    out = perpt(P0,P1,t)
    plt.plot(out[0],out[1],color=color,lw=lw,zorder=z)
    return out
    
# Similar but for cublic splines
def bezierCurveCubic(A,B,r1=1,r2=1,color="black",lw=2,z=0):
    
    t = np.linspace(0,1,50)
    
    if A[0] == B[0]:
        R1 = [A[0]+r1,midY(A,B)+r2]
        R2 = [A[0]+r1,midY(A,B)-r2]
        
    if A[1] == B[1]:
        R1 = [midX(A,B)+r2,A[1]+r1]
        R2 = [midX(A,B)-r2,A[1]+r1]
        
    if A[0] != B[0] and A[1] != B[1]:
        
        # Midpoint to position the curve
        mdpt = midpt(A,B)
        # Angle between the points and the angle perpendicular to it
        ang1 = np.arctan2((A[1]-B[1]),(A[0]-B[0]))
        ang2 = ang1+np.pi/2
        
        # Shift along the connecting line
        sh = [r2*np.cos(ang1),r2*np.sin(ang1)]
        # Shift perpendicular to it
        r = [r1*np.cos(ang2)+mdpt[0],r1*np.sin(ang2)+mdpt[1]]
        
        R1 = [r[0]+sh[0],r[1]+sh[1]]
        R2 = [r[0]-sh[0],r[1]-sh[1]]
        
    P0 = perpt(A,R1,t)
    P1 = perpt(R1,R2,t)
    P2 = perpt(R2,B,t)
    
    Q0 = perpt(P0,P1,t)
    Q1 = perpt(P1,P2,t)
    
    out = perpt(Q0,Q1,t)
    plt.plot(out[0],out[1],color=color,lw=lw,zorder=z)
    return out

# Arc drawing functions
# List of coordinates
def arc(xy,r,th=[0,np.pi],n=100):
    x = np.cos(np.linspace(th[0],th[1],n+1))*r+xy[0]
    y = np.sin(np.linspace(th[0],th[1],n+1))*r+xy[1]
    x = x[:n]
    y = y[:n]
    return [[a,b] for a,b in zip(x,y)]

# List of x positions and list of y positions
def arcXY(xy,r,th=[0,np.pi],n=100):
    x = np.cos(np.linspace(th[0],th[1],n+1))*r+xy[0]
    y = np.sin(np.linspace(th[0],th[1],n+1))*r+xy[1]
    x = x[:n]
    y = y[:n]
    return [x,y]

###############################################################################
###############################################################################
##
## DISTANCE AND POSITIONING FUNCTIONS
##
###############################################################################
###############################################################################
    
# Euclidean distance between nodes or points
def dist(A,B):
    p1 = (A[0]-B[0])**2
    p2 = (A[1]-B[1])**2
    return np.sqrt(p1+p2)


# Find midpoints between nodes in two dimensions
def midpt(A,B):
    x = (A[0] + B[0])/2
    y = (A[1] + B[1])/2
    return [x,y]

# Just the x or y coordinate if that's easier
def midX(A,B):
    return (A[0] + B[0])/2

def midY(A,B):
    return (A[1] + B[1])/2

# Find the point that is some percentage of the way between A and B
# 0 is at the center of A, 1 is at the center of B
def perpt(A,B,p):
    x = (A[0])*(1-p) + (B[0])*(p)
    y = (A[1])*(1-p) + (B[1])*(p)
    return [x,y]

# Find the point that is some distance from B along the line between A and B
def distpt(A,B,d):
    dd = dist(A,B)
    if dd == 0:
        return A
    p = (dd-d)/(dd)
    return perpt(A,B,p)

###############################################################################
###############################################################################
##
## GRAPH REPRESENTATION
##
###############################################################################
###############################################################################

# Creates a dictionary of out-edges based on either a graph or a matrix
def edgeDict(G):
    
    edges = dict()
    if type(G) == Graph:
        N = G.size
        for i in range(N):
            edges[str(i)] = []
            for j in range(N):
                if G.Mat[i,j] != 0:
                    edges[str(i)] += [j]
        return edges
    elif issqmat(G):
        N = np.shape(G)[0]
        for i in range(N):
            edges[str(i)] = []
            for j in range(N):
                if G[i,j] != 0:
                    edges[str(i)] += [j]
        return edges
    else:
        raise ValueError("Input must be Graph object or ndarray")
        
# Convert a dictionary of edges into an adjacency matrix
def MatFromDict(D):
    R = np.zeros([len(D),len(D)])
    for key, value in D.items():
        R[int(key),value] = 1
    return R

###############################################################################
###############################################################################
##
## GRAPH THEORETIC PROPERTIES
##
###############################################################################
###############################################################################

# A randomized adjacency matrix
def randAjdMat(N=5,directed=True,prob=.2):
    R = np.zeros([N,N],dtype="int")
    
    if directed == True:
        for x in range(N):
            for y in range(N):
                R[x,y] = np.random.choice([0,1],p=[1-prob,prob])
    if directed == False:
        for x in range(N):
            for y in range(x):
                r = np.random.choice([0,1],p=[1-prob,prob])
                R[x,y],R[y,x] = r
                
    return R


# Create a graph where every vertex has the same number of neighbors
def regularGraph(N=5,d=2,lim=100):
    ## Can't connect to more verticies than exist
    if d > (N-1):
        raise ValueError("Degree of a regular graph must be less than its size.")
    ## The number of vertices of odd degree must be even
    if N % 2 == 1 and d % 2 == 1:
        raise ValueError("No such graph due to handshaking lemma.")
    ctr = 0
    cr = False
    while cr == False:
        ctr += 1
        if ctr > lim:
            # Prevent function from running too long if accidentally given a 
            # excessive input
            raise ValueError("Unable to find matching graph.")
        cr = True
        # Create a possibly regular graph. Retry if it isn't regular.
        R = genregmat(N,d)
        for i in R.values():
            if len(i) != d:
                cr = False
                break
    #print(ctr)
    return MatFromDict(R)

# Method for generating an adjacency matrix that is fairly likely to be 
# regular. Simply picks random connections and fills in a dictionary with them.
def genregmat(N=5,d=2):
    S = [i for i in range(0,N)]
    D = {}
    for i in range(N):
        D[str(i)] = []
    for i in range(N):
        if len(S) == 0:
            break
        l = len(D[str(i)])
        if l < d:

            S = [x for x in S if x != i]
            n = min(d-l,len(S))
            r = random.sample(S,n)
            for v in r:
                D[str(i)].append(v)
                if i not in D[str(v)]:
                    D[str(v)].append(i)
            for pos, val in enumerate(S):
                if len(D[str(val)]) == d:
                    del S[pos]

    return D

def connectedGraph(N,directed=True):
    while True:
        t = randAjdMat(N,directed=directed)
        if len(connected(t,0)) == N:
            return t
        

# Make a copy of an adjacency matrix modified so that every edge is bidirectional
def undirected(R):
    M = R.copy()
    if issqmat(R):
        N = np.shape(R)[0]
        for x in range(N):
            for y in range(x):
                m = max(M[x,y], M[y,x])
                M[x,y], M[y,x] = m,m        
    return M

# Check if a graph has edges in both directions
def isUndir(G):
    if issqmat(G):
        N = np.shape(G)[0]
        for x in range(N):
            for y in range(x):
                if G[x,y] != G[y,x]:
                    return False
    return True

# Check if a Graph pobject or a relation matrix is cylic
def checkCyclic(R):
    if type(R) == Graph:
        A = R.Mat.copy()
        for i in range(len(A)):
            A[i,i] = 0
        emptyRow = True
        while emptyRow == True:
            n = len(A)
            if n == 0:
                return False
            for i in range(n):
                if sum(A[i,]) == 0:
                    A = np.delete(A,i,0)
                    A = np.delete(A,i,1)
                    emptyRow = True
                    break
                emptyRow = False
        return True
    
    elif issqmat(R):
        A = R.copy()
        for i in range(len(A)):
            A[i,i] = 0
        emptyRow = True
        while emptyRow == True:
            n = len(A)
            if n == 0:
                return False
            for i in range(n):
                if sum(A[i,]) == 0:
                    A = np.delete(A,i,0)
                    A = np.delete(A,i,1)
                    emptyRow = True
                    break
                emptyRow = False
        return True

# Extract a subgraph
def subgraph(R,L):
    if issqmat(R):
        t = R.copy()
        t = t[L,:]
        t = t[:,L]
        return t


# Explore the adjacency matrix in a depth first search
def dfs(R,x):
    checked = []
    working = [x]
    d = edgeDict(R)
    depth = [np.NaN]*len(d)
    while len(working) > 0:
        cur = working[-1]
        new = False
        for i in d[str(cur)]:
            if i not in checked and i not in working:
                working.append(i)
                new = True
                break
        if new == False:
            checked.append(working.pop())
            depth[cur] = len(working)

    return depth


# Use a breadth first search to find everything connected to vertex x.
def connected(R,x):
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
    while ctr < n:
        if ctr in L:
            ctr += 1
            continue
        else:
            T = connected(R,ctr)
            out.append(T)
            L += T
            ctr += 1
    return out


# Degree of each vertex, IN PROGRESS
def degreeUndir(G):
    R = undirected(G)
    N = np.shape(R)[0]
    deg = [0]*N
    for i in range(N):
        for j in range(i+1):
            if R[i,j] != 0:
                deg[i] += 1
                deg[j] += 1
    return deg
    
def outdegree(R):
    N = np.shape(R)[0]
    deg = [0]*N
    for i in range(N):
        for j in range(N):
            if R[i,j] != 0:
                deg[i] += 1

    return deg

def indegree(R):
    N = np.shape(R)[0]
    deg = [0]*N
    for i in range(N):
        for j in range(N):
            if R[i,j] != 0:
                deg[j] += 1
    return deg
                


###############################################################################
###############################################################################
##
## OTHER MISCELLANEOUS FUNCTIONS
##
###############################################################################
###############################################################################
    

# Find the minimum and maximum of a list
def minmax(L):
    return [min(L),max(L)]

# Check if input is a square numpy matrix
def issqmat(M):
    if type(M) == np.ndarray:
        s = np.shape(M)
        if s[0] != s[1]:
            raise ValueError("Adjacency matrix must be square")
        return True
    raise ValueError("Must be an ndarry object")
    
###############################################################################
###############################################################################
##
## EXAMPLE FUNCTIONS
##
###############################################################################
###############################################################################

# Arranges the elements of the graph in a circle and draws arrows between them
def connectogram(R,L=[None],title="",size=[7,7],nodeSize=.3,
                      nodeCol = (.53, .81, .94), lineSize  = 2, lineCol = "black"):
    
    fig, ax = makeCanvas(size=size)
    
    n = np.shape(R)[0]
    if len(L) != n:
        L = [str(i) for i in range(n)]

    xy = arc((0,0),2.5,[0,np.pi*2],n)
    G = Graph(rdef=nodeSize,tscaledef=1.5,coldef = nodeCol)
    
    for i,p in enumerate(xy):
        G.addNode(p,text=str([L[i]][0]),z=2)

    G.Mat = R
    G.drawNodes(fig,ax)
    G.drawArrows(col=lineCol,wd=lineSize)
    G.drawText(fig,ax)
    plt.title(title)
    return G, fig, ax

# Arranges the elements of the graph in a circle and draws lines between them
# Option to curve the connections
def connectogramUndir(R,L=[None],title="",size=[7,7],curve=0,nodeSize=.3,
                      nodeCol = (.53, .81, .94), lineSize  = 2, lineCol = "black"):
    
    fig, ax = makeCanvas(size=size)
    
    n = np.shape(R)[0]
    if len(L) != n:
        L = [str(i) for i in range(n)]

    for x in range(n):
        for y in range(n):
            if R[y,x] != 0:
                R[x,y] = R[y,x]
                R[y,x] = 0

    xy = arc((0,0),2.5,[0,np.pi*2],n)
    G = Graph(rdef=nodeSize,tscaledef=1.5,coldef = nodeCol)
    
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str([L[i]][0]),z=2)

    G.Mat = R
    G.drawNodes(fig,ax)
    G.drawText(fig,ax)
    if curve == 0:
        G.drawLines(col=lineCol)
    if curve != 0:
        for i in np.argwhere(R != 0):
            if i[0] == i[1]:
                continue
            if abs(i[0] - i[1]) >= (n//2):
                bezierCurve(G.pos[i[0]],G.pos[i[1]],r=-curve,color=lineCol)
            else:
                bezierCurve(G.pos[i[0]],G.pos[i[1]],r=curve,color=lineCol)
    plt.title(title)

    return G, fig, ax

###############################################################################
###############################################################################
##
## TESTING FUNCTIONALITY
##
###############################################################################
###############################################################################
def test():

    # Make a graph
    G = Graph(rdef=.5)

    # Add a node with all default properties
    G.addNode()
    # Add a node at a different position with text
    G.addNode([-2,.5],text="Hello")

    # Add a node at a different position with different size
    G.addNode([1,-1.5],r=.3)
    
    # Create some edges
    G.addEdges([0,1],[1,2])
    
    # Change an existing node
    G.colors[0] = 'red'
    
    print(G.Mat)
    print(type(G.Mat))
    
    fig, ax = makeCanvas()
    G.QuickDraw(fig,ax)
    G.drawText(fig,ax)
    
    
    # Use the simplified cubie bezier curve functions
    bezierCurve(G.pos[0],G.pos[1],2)
    bezierCurveCubic(G.pos[0],G.pos[2],-2,4)
    
    # Connecting arrow between points
    connectArr([0,0],[1.1,2])
    
    #Connecting arrow between nodes
    connectArr(G.pos[0],G.pos[1],headpos=.5)
    loop(G.pos[0],th=.1,rot=.5)
    
    angles = np.arange(0, 360 + 45, 45)
    ells = [patches.Ellipse((1, 1), 4, 2, a,zorder=0) for a in angles]
    for e in ells:
        e.set_alpha(0.1)
        ax.add_patch(e)
    
    fig2, ax2 = makeCanvas(size=[3,3])
    G.QuickDraw(fig2,ax2)
    G.drawText(fig2,ax2)
    
    fig3, ax3 = makeCanvas(size=[5,5])
    
    G1 = Graph(rdef=.3)
    ps = [[0,0],[0,1],[2,0],[1,-1]]
    for i in range(len(ps)):
        G1.addNode(ps[i],text=str(i))
    
    G1.addEdgesBi([2,2,2,1],[0,1,3,1])
    
    G1.QuickDraw(fig3,ax3)
    loop(G1.pos[1],r=.3,th=1.2,rot=.5)
    

#test()