from Graphs import *

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
    G.drawLoops()
    G.Mat = R
    G.drawNodes()
    G.drawArrows(term=.32)
    plt.title(title)