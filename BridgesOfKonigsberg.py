import GraphNoAnim

def EulersBridges():
    # There are four landmasses and seven bridges with the following layout

    G = Graph(rdef=.3)
    
    G.addNode([-2,0])
    G.addNode([2,0])
    G.addNode([0,2])
    G.addNode([0,-2])
    

    G.addEdges([0,1,1],[1,2,3])
    
    connectArc(G.Nodes[0],G.Nodes[2],3)
    connectArc(G.Nodes[0],G.Nodes[2],3,True)
    connectArc(G.Nodes[0],G.Nodes[3],3)
    connectArc(G.Nodes[0],G.Nodes[3],3,True)
    G.QuickDraw()

EulersBridges()