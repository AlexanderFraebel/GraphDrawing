import Graphs
import DijkstrasAlgorithm
import numpy as np


# Greatest number of steps needed to get from each vertex to any other.
# That is starting at node N it never take more than ecc(N) steps to reach any
# other node in the graph.
def eccentricity(R):
    L = []
    if issqmat(R):
        for i in range(np.shape(R)[0]):
            di = dijkstra(R,i)
            L.append(max([len(x) for x in di[1]]))
    if max(L) == np.inf:
        print("Not a totally connected graph")
    return L

# Minimum eccentricity
def radius(R):
    L = eccentricity(R)
    return min(L)

# Maximum eccentricity
def diameter(R):
    L = eccentricity(R)
    return max(L)

R = regularGraph(10,2)
G = connectogramUndir(R)
print("Eccentricity of Each Node".format(eccentricity(R)))
print(radius(R))
print(diameter(R))