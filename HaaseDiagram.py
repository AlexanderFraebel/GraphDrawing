import numpy as np
from Graphs import *

def primeFactors(n):
    if n == 1:
        return 0
    factors = 0
    d = 2
    while(n > 1):
        while(n % d == 0):
            factors += 1
            n //= d
        d = d + 1
        if(d*d > n):
            if(n > 1): 
                factors += 1
            break
    return factors


def HaaseDivisibility(L,title=None,H=None,sc=.4,ptsz=.05,
                  drawConnections=True):
    
    L = sorted(L)
    
    numfacts = [primeFactors(i) for i in L]
    m = max(numfacts)
    lvls = [list() for i in range(m+1)]
    
    for i in L:
        n = primeFactors(i)
        lvls[n].append(i)
    
    G = Graph(rdef=ptsz,tscaledef=5)
    fig,ax = makeCanvas([-.4,1.4],[-.1,sc*len(lvls)],[12,12])

    y = 0
    ## We'll use these to store some information about the previous level
    XYold = []
    XYnew = []
    XYout = []
    for f in lvls:
        ## The XY positions of the previous level are now old
        ## We also reset the XY positions for this level to be blank so we can
        ## start fillign them in.
        XYold = XYnew
        XYnew = []
        ## Place the text on top of a circle. If the point is a factor of the
        ## subgraph make it salmon colored. Otherwise light blue.
        ## Also take note of the current position.
        x = np.linspace(0,1,len(f))
        if len(f) == 1:
            x = [.5]
        for l in range(len(f)):
            
            if H != None and H % f[l] == 0:
                G.addNode([x[l],y],text=f[l],col='#FF91A4')
            else:
                G.addNode([x[l],y],text=f[l])
            XYnew.append((f[l],x[l],y))
            XYout.append((x[l],y))
        
        ## Check if any of the current numbers are divisible by the ones in the
        ## level before. If they are draw a connecting line.
        ## Then also check if both numbers are factors of the subgraph. If they
        ## are then make the line red, otherwise make it gray. Gray lines are
        ## drawn beneath the red lines and are slightly thinner.
        if drawConnections == True:
            for val,xpos,ypos in XYnew:
                for valOLD,xposOLD,yposOLD in XYold:
                    if val % valOLD == 0:
                        if H != None and H % val == 0 and H % valOLD == 0:
                            connect([xpos,ypos],[xposOLD,yposOLD],width=2,col='red')
                        else: 
                            connect([xpos,ypos],[xposOLD,yposOLD],width=2)
        
        y += sc
    G.QuickDraw(fig,ax)
    G.drawText(fig,ax)
    if title != None:
        plt.savefig(title,dpi=100)
    print(lvls)
    #print(XYout)
    return XYout

HaaseDivisibility([1,2,3,4,5,6,9,10,12,15,18,20,30,60],H=18,ptsz=.1)