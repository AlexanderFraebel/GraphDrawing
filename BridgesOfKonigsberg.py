from GraphNoAnim import *

def EulersBridges():
    # There are four landmasses and seven bridges with the following layout

    G = Graph(rdef=.3)
    
    G.addNode([-2,0])
    G.addNode([2,0])
    G.addNode([0,2])
    G.addNode([0,-2])
    

    G.addEdges([0,1,1],[1,2,3])
    
    bezierCurve(G.Nodes[0],G.Nodes[2],1)
    bezierCurve(G.Nodes[0],G.Nodes[2],-1)
    bezierCurve(G.Nodes[0],G.Nodes[3],1)
    bezierCurve(G.Nodes[0],G.Nodes[3],-1)
    G.QuickDraw()

EulersBridges()