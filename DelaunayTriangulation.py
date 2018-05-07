from Graphs import *
import numpy as np
from ParticularGraphs import flowerSnarkGraph

# Is D inside the cirle discribed by the points A,B,C
def inCircle(A,B,C,D):
    f = 2*(A[0]*(B[1]-C[1])+B[0]*(C[1]-A[1])+C[0]*(A[1]-B[1]))
    x = ((A[0]*A[0]+A[1]*A[1])*(B[1]-C[1])+
         (B[0]*B[0]+B[1]*B[1])*(C[1]-A[1])+
         (C[0]*C[0]+C[1]*C[1])*(A[1]-B[1]) )/f
    y = ((A[0]*A[0]+A[1]*A[1])*(C[0]-B[0])+
         (B[0]*B[0]+B[1]*B[1])*(A[0]-C[0])+
         (C[0]*C[0]+C[1]*C[1])*(B[0]-A[0]) )/f
    if dist([x,y],A) > dist([x,y],D):
        return True
    return False
    

G = flowerSnarkGraph()
makeCanvas()

G.colors[10] = 'red'
G.colors[14] = 'red'
G.colors[19] = 'red'
G.colors[0] = 'red'
G.QuickDraw()
G.drawText()
print(inCircle(G.pos[10],G.pos[14],G.pos[19],G.pos[0]))

# while len(stk) > 0:
#   ab = stk.pop()
#   if ab is illegal:
#       dlip ab to cd
#