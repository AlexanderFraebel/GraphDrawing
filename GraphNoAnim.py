import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

class Graph:
    def __init__(self,xlims=[-3,3],ylims=[-3,3],size=[7,7],
                 rdef=.5,tscaledef=30,coldef=(.53, .81, .94)):

        ## Setup the plot area
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0], size[1])
        self.ax = plt.axes(xlim=xlims, ylim=ylims)
        self.ax.axis('off')
        
        # Set a few default characteristics
        self.rdef = rdef
        self.tscaledef = tscaledef
        self.coldef = coldef
        
        # Prepare a list of nodes
        self.Nodes = []
        
        # Prepare a matrix of connections
        self.Mat = [0]
    
    ## Create new node
    def addNode(self,xy=[0,0],r=None,col=[None],text="",tscale=None,z=1):
        if r == None:
            r = self.rdef
        if tscale == None:
            tscale = self.tscaledef
        if any(i == None for i in col):
            col = self.coldef
        self.Nodes.append(Node(xy,r,col,text,tscale,z))

        t = np.zeros((len(self.Nodes),len(self.Nodes)))
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
        for i in self.Nodes:
            self.ax.add_patch(i.circ)
            plt.text(i.xy[0],i.xy[1],i.text,size=i.r*i.tscale,
                 ha='center',va='center',zorder=i.z)
        for i in np.argwhere(self.Mat != 0):
            ter = distpt(self.Nodes[i[0]],self.Nodes[i[1]],.32)
            plt.plot([self.Nodes[i[0]].x,ter[0]],[self.Nodes[i[0]].y,ter[1]],
                     color="black",lw=2,zorder=0)

    def drawNodes(self):
        for i in self.Nodes:
            self.ax.add_patch(i.circ)
            plt.text(i.xy[0],i.xy[1],i.text,size=i.r*i.tscale,
                 ha='center',va='center',zorder=i.z)
    
    def drawArrows(self,col="black",wd=2,hwd=.1,hln=.2,term=.3):
        for i in np.argwhere(self.Mat != 0):
            if i[0] == i[1]:
                continue
            ter = distpt(self.Nodes[i[0]],self.Nodes[i[1]],term)
            connectArrPts(self.Nodes[i[0]].x,self.Nodes[i[0]].y,ter[0],ter[1],
                          width=wd,headwidth=hwd,headlength=hln,z=0,col=col)
            
    def drawLines(self,col="black",wd=2):
        for i in np.argwhere(self.Mat != 0):
            if i[0] == i[1]:
                continue
            ter = dist(self.Nodes[i[0]],self.Nodes[i[1]])
            plt.plot([self.Nodes[i[0]].x,ter[0]],[self.Nodes[i[0]].y,ter[1]],
                     color=col,lw=wd,zorder=0)
            
            
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
    

# Draw straight line connections between nodes
def connect(A,B,col="black",width=1,z=0):
    plt.plot([A.x,B.x],[A.y,B.y],color=col,lw=width,zorder=z)

def connectArr(A,B,col="black",width=1,headwidth=.2,headlength=.2,z=0):
    plt.arrow(A.x,A.y,B.x-A.x,B.y-A.y,color=col,lw=width,zorder=z,
              head_width=headwidth, head_length=headlength,
              length_includes_head=True)

# Draw straight line connections between arbitrary points
def connectPts(x1,y1,x2,y2,col="black",width=1,z=0):
    plt.plot([x1,x2],[y1,y2],color=col,lw=width,zorder=z)
    
def connectArrPts(x1,y1,x2,y2,col="black",width=1,headwidth=.2,headlength=.2,z=0):
    plt.arrow(x1,y1,x2-x1,y2-y1,color=col,lw=width,zorder=z,
          head_width=headwidth, head_length=headlength,
          length_includes_head=True)


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

# Distance functions
def dist(A,B):
    p1 = (A.x-B.x)**2
    p2 = (A.y-B.y)**2
    return np.sqrt(p1+p2)

def distMat(L):
    ps = [(i.x,i.y) for i in L]
    return distance.squareform(np.round(distance.pdist(ps),3))


# Find midpoints of nodes in two dimensions or in one
def midpt(A,B):
    x = (A.x + B.x)/2
    y = (A.y + B.y)/2
    return [x,y]

def midX(A,B):
    return (A.x + B.x)/2

def midY(A,B):
    return (A.y + B.y)/2

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
        return A.xy
    p = (dd-d)/(dd)
    return perpt(A,B,p)


# Find the minimum and maximum of a list
def minmax(L):
    return [min(L),max(L)]

# Connection plot
def connectogram(R,L=[None],title="",size=[7,7]):
    
    n = R.shape[0]
    if len(L) != n:
        L = [str(i+1) for i in range(n)]
    
    print(R)
    xy = arc((0,0),2.5,[0,np.pi*2],n)
    G = Graph(rdef=.3,tscaledef=70,size=size)
    
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str([L[i]][0]),z=2)
    
    G.Mat = R
    G.drawNodes()
    G.drawArrows(term=.32)
    plt.title(title)

## Test of various functionality
def test():

    G = Graph([-3,3],[-3,3],[7,7],rdef=.5)
    G.addNode()
    print(G.Mat)
    G.addNode([-2,.5],text="Hello")
    print(G.Mat)
    
    G.addNode([1,-1.5])
    print(G.Mat)
    G.addEdges([0,1],[1,2])
    print(G.Mat)
    G.Nodes[0].update(col='red')
    G.QuickDraw()
    
    bezierCurve(G.Nodes[0],G.Nodes[1],2)
    
    bezierCurveCubic(G.Nodes[0],G.Nodes[2],-2,4)
#test()
