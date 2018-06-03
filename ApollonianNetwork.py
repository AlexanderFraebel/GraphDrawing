from Graphs import *

# https://lsandig.org/blog/2014/08/apollon-python/en/

class Circle:
    def __init__(self,pos,r,num):
        self.pos = pos + 0j
        self.r = r
        self.cur = 1/r
        self.num = num
        

def outerCirc(A,B,C):
    c4 = -2*np.sqrt(A.cur*B.cur + B.cur*C.cur + A.cur*C.cur) + A.cur + B.cur + C.cur
    m4 = (-2 * np.sqrt(A.cur*A.pos*B.cur*B.pos + 
                       A.cur*A.pos*C.cur*C.pos + 
                       B.cur*B.pos*C.cur*C.pos) + A.cur*A.pos + B.cur*B.pos + C.cur*C.pos)/c4
    
    return Circle(m4,1/c4,0)


def tanCircFromRad(a,b,c,G):
    C2 = Circle(0,a,1)
    C3 = Circle(a+b,b,2)
    
    p4x = (a*a + a*b + a*c - b*c)/(a+b)
    p4y = np.sqrt((a+c)*(a+c) - p4x*p4x)
    
    C4 = Circle(p4x + p4y*1j,c,3)
    C1 = outerCirc(C2,C3,C4)
    
    ## Center the cirlces
    C2.pos -= C1.pos
    C3.pos -= C1.pos
    C4.pos -= C1.pos
    C1.pos -= C1.pos
    
    return C1,C2,C3,C4


def secondSol(F,C1,C2,C3,G):
    curn = 2*(C1.cur+C2.cur+C3.cur) - F.cur
    posn = (2*(C1.cur*C1.pos+C2.cur*C2.pos+C3.cur*C3.pos) - F.cur*F.pos)/curn
    Cn = Circle(posn,1/curn,G.size)
    return Cn

    
def apolloRecur(a,b,c,d,lim,itr,G):


    if a.r > 0 and b.r > 0:
        G.addEdges([a.num],[b.num])
    if a.r > 0 and c.r > 0:
        G.addEdges([a.num],[c.num])
    if a.r > 0 and d.r > 0:
        G.addEdges([a.num],[d.num])

            
    if itr == 0:
        e0 = secondSol(a,b,c,d,G)
        G.addNode([e0.pos.real,e0.pos.imag])
        apolloRecur(e0,b,c,d,lim,itr+1,G)
    
    
    e1 = secondSol(b,a,c,d,G)
    if e1.cur < lim:
        G.addNode([e1.pos.real,e1.pos.imag])
        apolloRecur(e1,a,c,d,lim,itr+1,G)
    
    
    e2 = secondSol(c,a,b,d,G)
    if e2.cur < lim:
        G.addNode([e2.pos.real,e2.pos.imag])
        apolloRecur(e2,a,b,d,lim,itr+1,G)
        
        
    e3 = secondSol(d,a,b,c,G)
    if e3.cur < lim:
        G.addNode([e3.pos.real,e3.pos.imag])
        apolloRecur(e3,a,b,c,lim,itr+1,G)
      
    

def ApollonianGasket(A,B,C,lim=100,**kwargs):
    G = Graph(**kwargs)
    
    a,b,c,d = tanCircFromRad(A,B,C,G)


    for i in [a,b,c,d]:
        G.addNode([i.pos.real,i.pos.imag])

    G.addEdges([1,2,3],[2,3,1])

    apolloRecur(a,b,c,d,lim,0,G)
    G.delNode(0)
    return G
    
makeCanvas([-2.2,2.2],[-2.2,2.2],size=[16,16])

N = 30
G = ApollonianGasket(1,1,1,N,NodeSize=.04,NodeColor='gray')

G.drawNodes()
G.drawLines()
#G.drawText()