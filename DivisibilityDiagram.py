from Graphs import *

L = [2,3,5,10,15,30]

## Relational diagram is set to 1 if i divides j and zero otherwise
R = np.zeros((len(L),len(L)))
for x,i in enumerate(L):
    for y,j in enumerate(L):
        if i != j and i % j == 0:
            R[x,y] = 1


connectogram(R,L)