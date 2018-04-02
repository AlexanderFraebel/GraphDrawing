from GraphNoAnim import *

L = [2,3,5,10,15,30]

## Relational diagram is set to 1 if i divides j and zero otherwise
R = np.zeros((len(L),len(L)))
for x,i in enumerate(L):
    for y,j in enumerate(L):
        if i != j and i % j == 0:
            R[x,y] = 1


            
def connectogram(L,R):
    print(R)
    n = len(L)
    xy = arc((0,0),2.5,[0,np.pi*2],n)
    G = Graph(rdef=.3,tscaledef=70)
    
    for i,pos in enumerate(xy):
        G.addNode(pos,text=str([L[i]][0]),z=2)
        
    G.QuickDraw()
    
    for i in np.argwhere(R==1):
        A = G.Nodes[i[0]]
        B = G.Nodes[i[1]]
        ter = distpt(A,B,.32)
        connectArrPts(A.x,A.y,ter[0],ter[1],width=1.5,headwidth=.1,headlength=.2)


connectogram(L,R)