import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.patches as patches

class Graph:
    def __init__(self,xlims=[-3,3],ylims=[-3,3],size=[7,7],
                 rdef=.5,tscaledef=10,coldef=(.53, .81, .94)):

        ## Setup the plot area
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0], size[1])
        self.ax = plt.axes(xlim=xlims, ylim=ylims)
        self.ax.axis('off')
        self.size = size
        
        # Set a few default characteristics
        self.rdef = rdef
        self.tscaledef = tscaledef
        self.coldef = coldef
        
        # Prepare a list of nodes
        self.Nodes = []
        
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
        self.Nodes.append(Node(xy,r,col,text,tscale,z))

        # Expand the adjacency matrix
        t = np.zeros((len(self.Nodes),len(self.Nodes)))
        t[:-1,:-1] = self.Mat
        self.Mat = t
        
    def resize(self,size=[7,7]):
        self.size = size
        self.fig.set_size_inches(size[0], size[1])

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
        del self.Nodes[n]
        self.Mat = np.delete(self.Mat,n,0)
        self.Mat = np.delete(self.Mat,n,1)
    
    def delEdges(self,A,B):
        self.Mat[A,B] = 0
    
    def delEdgesBi(self,A,B):
        self.Mat[A,B] = 0
        self.Mat[B,A] = 0

    # Simple drawing functions for common situations
    def QuickDraw(self):
        d = np.sqrt(self.size[0]**2+self.size[1]**2)/2
        for i in self.Nodes:
            self.ax.add_patch(i.circ)
            plt.text(i.xy[0],i.xy[1],i.text,size=i.r*i.tscale*d,
                 ha='center',va='center',zorder=i.z)
        for i in np.argwhere(self.Mat != 0):
            connect(self.Nodes[i[0]],self.Nodes[i[1]])

    def drawNodes(self):
        d = np.sqrt(self.size[0]**2+self.size[1]**2)/2
        for i in self.Nodes:
            self.ax.add_patch(i.circ)
            plt.text(i.xy[0],i.xy[1],i.text,size=i.r*i.tscale*d,
                 ha='center',va='center',zorder=i.z)
    
    def drawArrows(self,col="black",wd=2,hwd=.1,hln=.2,term=None):
        for i in np.argwhere(self.Mat != 0):
            if i[0] == i[1]:
                continue
            connectArr(self.Nodes[i[0]],self.Nodes[i[1]],
                          width=wd,headwidth=hwd,headlength=hln,z=0,col=col)
    
    def drawLines(self,col="black",wd=2):
        for i in np.argwhere(self.Mat != 0):
            if i[0] == i[1]:
                continue
            connect(self.Nodes[i[0]],self.Nodes[i[1]],col=col,width=wd)
            
            
## The Node class has the properties of each vertex of the graph.
class Node:
    def __init__(self,xy=[0,0],r=.5,col=(.53, .81, .94),text="",tscale=30,z=1):
        self.xy = xy
        self.x = xy[0]
        self.y = xy[1]
        self.r = r
        self.col = col
        self.z = z
        self.text = text
        self.tscale = tscale
        self.circ = plt.Circle(self.xy,radius = self.r, fc = self.col,zorder=self.z)

    def update(self,xy=None,r=None,col=[None],text=None,tscale=None,z=None):
        if xy != None:
            self.xy = xy
            self.x = xy[0]
            self.y = xy[1]
        if r != None:
            self.r = r
        if any(i == None for i in col):
            pass
        else:
            self.col = col
        if z != None:
            self.z = z
        if text != None:
            self.text = text
        if tscale != None:
            self.tscale = tscale

        self.circ = plt.Circle(self.xy,radius = self.r, fc = self.col,zorder=self.z)


###############################################################################
###############################################################################
##
## DRAWING CONNECTING LINES
##
###############################################################################
###############################################################################

# Draw straight line connection between nodes or points
def connect(A,B,col="black",width=1,z=0):
    if type(A) == Node and type(B) == Node:
        plt.plot([A.x,B.x],[A.y,B.y],color=col,lw=width,zorder=z)
    elif type(A) == list and type(B) == list:
        if len(A) == 2 and len(B) == 2:
            plt.plot([A[0],B[0]],[A[1],B[1]],color=col,lw=width,zorder=z)
        else:
            raise ValueError("A and B must length 2 if given as lists")    
    elif type(A) == np.ndarray and type(B) == np.ndarray:
        if len(A) == 2 and len(B) == 2:
            plt.plot([A[0],B[0]],[A[1],B[1]],color=col,lw=width,zorder=z)
        else:
            raise ValueError("A and B must length 2 if given as ndarray")
    else:
        raise ValueError("A and B do not match or are not recognized")

# Draw straight line arrow between nodes or points
# When connecting nodes automatically stops the arrow at the edge of the node
def connectArr(A,B,col="black",width=1,headwidth=.2,headlength=.2,z=0):
    if type(A) == Node and type(B) == Node:
        ter = distpt(A,B,B.r)
        plt.arrow(A.x,A.y,ter[0]-A.x,ter[1]-A.y,color=col,lw=width,zorder=z,
                  head_width=headwidth, head_length=headlength,
                  length_includes_head=True)
    elif type(A) == list and type(B) == list:
        if len(A) == 2 and len(B) == 2:
            plt.arrow(A[0],A[1],B[0]-A[0],B[1]-A[1],color=col,lw=width,zorder=z,
                  head_width=headwidth, head_length=headlength,
                  length_includes_head=True)
        else:
            raise ValueError("A and B must length 2 if given as lists")  
    elif type(A) == np.ndarray and type(B) == np.ndarray:
        if len(A) == 2 and len(B) == 2:
            plt.arrow(A[0],A[1],B[0]-A[0],B[1]-A[1],color=col,lw=width,zorder=z,
                  head_width=headwidth, head_length=headlength,
                  length_includes_head=True)
        else:
            raise ValueError("A and B must length 2 if given as ndarray")
    else:
        raise ValueError("A and B do not match or are not recognized")

            
# Connection to self
# Creates a circle the intersects the center of the node
# th controls the position of the circle and rot controls the position of the
# arrow around the loop
def loop(A,r=.4,th=0,rot=0,col="black",width=1,headwidth=.2,headlength=.2,z=0):
    a = np.linspace(0,np.pi*2)
    x = r*np.cos(a+rot)+A.x+(np.cos(th)*r)
    y = r*np.sin(a+rot)+A.y+(np.sin(th)*r)
    plt.plot(x,y,color=col,lw=width,zorder=z)
    connectArr([x[0],y[0]],[x[1],y[1]],headwidth=.2,headlength=.2,z=z)


###############################################################################
###############################################################################
##
## BEZIER CURVES
##
###############################################################################
###############################################################################

# Create arbitrary Bezier splines with either one or two control points
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
def bezierCurve(A,B,r=1):
    t = np.linspace(0,1,50)
    mdpt = midpt(A,B)
    if A.x == B.x:
        R = Node([A.x+r,midY(A,B)])
    if A.y == B.y:
        R = Node([midX(A,B),A.y+r])
    if A.x != B.x and A.y != B.y:
        ang = np.arctan2((A.y-B.y),(A.x-B.x))+np.pi/2
        R = Node([r*np.cos(ang)+mdpt[0],r*np.sin(ang)+mdpt[1]])
    P0 = perpt(A,R,t)
    P1 = perpt(R,B,t)
    
    out = perpt(P0,P1,t)
    plt.plot(out[0],out[1],color="black",lw=2,zorder=0)
    return out
    
# Similar but for cublic splines
def bezierCurveCubic(A,B,r1=1,r2=1):
    
    t = np.linspace(0,1,50)
    
    if A.x == B.x:
        R1 = Node([A.x+r1,midY(A,B)+r2])
        R2 = Node([A.x+r1,midY(A,B)-r2])
        
    if A.y == B.y:
        R1 = Node([midX(A,B)+r2,A.y+r1])
        R2 = Node([midX(A,B)-r2,A.y+r1])
        
    if A.x != B.x and A.y != B.y:
        
        # Midpoint to position the curve
        mdpt = midpt(A,B)
        # Angle between the points and the angle perpendicular to it
        ang1 = np.arctan2((A.y-B.y),(A.x-B.x))
        ang2 = ang1+np.pi/2
        
        # Shift along the connecting line
        sh = [r2*np.cos(ang1),r2*np.sin(ang1)]
        # Shift perpendicular to it
        r = [r1*np.cos(ang2)+mdpt[0],r1*np.sin(ang2)+mdpt[1]]
        
        R1 = Node([r[0]+sh[0],r[1]+sh[1]])
        R2 = Node([r[0]-sh[0],r[1]-sh[1]])
        
    P0 = perpt(A,R1,t)
    P1 = perpt(R1,R2,t)
    P2 = perpt(R2,B,t)
    
    Q0 = perpt(P0,P1,t)
    Q1 = perpt(P1,P2,t)
    
    out = perpt(Q0,Q1,t)
    plt.plot(out[0],out[1],color="black",lw=2,zorder=0)
    return out

# Arc drawing functions
# List of coordinates
def arc(xy,r,th=[0,np.pi],n=100):
    x = np.cos(np.linspace(th[0],th[1],n+1))*r+xy[0]
    y = np.sin(np.linspace(th[0],th[1],n+1))*r+xy[1]
    x = x[:n]
    y = y[:n]
    return [(a,b) for (a,b) in zip(x,y)]

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
## GRAPH THEORETIC PROPERTIES
##
###############################################################################
###############################################################################
    
# Creates a dictionary of out-edges based on either a graph or a matrix
def edgeDict(G):
    
    edges = dict()
    if type(G) == Graph:
        N = len(G.Nodes)
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
        



# Make every edge in a graph into an undirected edge
def makeUndir(G):
    if type(G) == Graph:
        N = len(G.Nodes)
        for x in range(N):
            for y in range(x):
                m = max(G.Mat[x,y], G.Mat[y,x])
                G.Mat[x,y], G.Mat[y,x] = m,m
    elif issqmat(G):
        N = np.shape(G)[0]
        for x in range(N):
            for y in range(x):
                m = max(G[x,y], G[y,x])
                G[x,y], G[y,x] = m,m        

def isUndir(G):
    if type(G) == Graph:
        N = len(G.Nodes)
        for x in range(N):
            for y in range(x):
                if G.Mat[x,y] != G.Mat[y,x]:
                    return False
    elif issqmat(G):
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



###############################################################################
###############################################################################
##
## DISTANCE AND POSITIONING FUNCTIONS
##
###############################################################################
###############################################################################
    
# Euclidean distance between nodes or points
def dist(A,B):
    if type(A) == Node and type(B) == Node:
        p1 = (A.x-B.x)**2
        p2 = (A.y-B.y)**2
        return np.sqrt(p1+p2)
    else:
        p1 = (A[0]-B[0])**2
        p2 = (A[1]-B[1])**2
        return np.sqrt(p1+p2)


# Find midpoints between nodes in two dimensions
def midpt(A,B):
    if type(A) == Node and type(B) == Node:
        x = (A.x + B.x)/2
        y = (A.y + B.y)/2
        return [x,y]
    else:
        x = (A[0] + B[0])/2
        y = (A[1] + B[1])/2
        return [x,y]

# Just the x or y coordinate if that's easier
def midX(A,B):
    if type(A) == Node and type(B) == Node:
        return (A.x + B.x)/2
    else:
        return (A[0] + B[0])/2

def midY(A,B):
    if type(A) == Node and type(B) == Node:
        return (A.y + B.y)/2
    else:
        return (A[1] + B[1])/2

# Find the point that is some percentage of the way between A and B
# 0 is at the center of A, 1 is at the center of B
def perpt(A,B,p):
    if type(A) == Node and type(B) == Node:
        x = (A.x)*(1-p) + (B.x)*(p)
        y = (A.y)*(1-p) + (B.y)*(p)
        return [x,y]
    else:
        x = (A[0])*(1-p) + (B[0])*(p)
        y = (A[1])*(1-p) + (B[1])*(p)
        return [x,y]

# Find the point that is some distance from B along the line between A and B
def distpt(A,B,d):
    dd = dist(A,B)
    if dd == 0:
        if type(A) == Node and type(B) == Node:
            return A.xy
        else:
            return A
    p = (dd-d)/(dd)
    return perpt(A,B,p)


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
def connectogram(R,L=[None],title="",size=[7,7]):
    
    n = R.shape[0]
    if len(L) != n:
        L = [str(i) for i in range(n)]

    xy = arc((0,0),2.5,[0,np.pi*2],n)
    G = Graph(rdef=.3,tscaledef=15,size=size)
    
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str([L[i]][0]),z=2)

    G.Mat = R
    G.drawNodes()
    G.drawArrows(term=G.rdef*1.1)
    plt.title(title)
    return G

# Arranges the elements of the graph in a circle and draws lines between them
def connectogramUndir(R,L=[None],title="",size=[7,7],curve=False):
    
    n = R.shape[0]
    if len(L) != n:
        L = [str(i) for i in range(n)]

    for x in range(n):
        for y in range(n):
            if R[y,x] != 0:
                R[x,y] = R[y,x]
                R[y,x] = 0

    xy = arc((0,0),2.5,[0,np.pi*2],n)
    G = Graph(rdef=.3,tscaledef=15,size=size)
    
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str([L[i]][0]),z=2)

    G.Mat = R
    G.drawNodes()
    if curve == False:
        G.drawLines()
    if curve == True:
        for i in np.argwhere(R != 0):
            if i[0] == i[1]:
                continue
            if i[0] - i[1] > (n//2):
                bezierCurve(G.Nodes[i[0]],G.Nodes[i[1]],r=-1)
            else:
                bezierCurve(G.Nodes[i[0]],G.Nodes[i[1]],r=1)
    plt.title(title)
    makeUndir(G)
    return G
    
    
###############################################################################
###############################################################################
##
## TESTING FUNCTIONALITY
##
###############################################################################
###############################################################################
def test():

    # Make a graph
    G = Graph([-3,3],[-3,3],[7,7],rdef=.5)

    # Add a node with all default properties
    G.addNode()
    # Add a node at a different position with text
    G.addNode([-2,.5],text="Hello")

    # Add a node at a different position with different size
    G.addNode([1,-1.5],r=.3)
    
    # Create some edges
    G.addEdges([0,1],[1,2])
    
    # Change an existing node
    G.Nodes[0].update(col='red')
    
    print(G.Mat)
    print(type(G.Mat))
    
    # Use the quickdraw function
    G.QuickDraw()
    
    # Use the simplified bezier curve function
    bezierCurve(G.Nodes[0],G.Nodes[1],2)
    
    # Use the simplified cubie bezier curve function
    bezierCurveCubic(G.Nodes[0],G.Nodes[2],-2,4)
    
    # Connecting arrow between points
    connectArr([0,0],[1.1,2])
    
    #Connecting arrow between nodes
    connectArr(G.Nodes[0],G.Nodes[2])
    
    #Make a loop
    loop(G.Nodes[0],th=.1,rot=.5)
    
    ## Make changes to the plot area by referring to the .ax property
    angles = np.arange(0, 360 + 45, 45)
    ells = [patches.Ellipse((1, 1), 4, 2, a,zorder=0) for a in angles]
    for e in ells:
        e.set_alpha(0.1)
        G.ax.add_patch(e)
        
    ## Resize after the fact
    #G.resize([12,12])
#test()
