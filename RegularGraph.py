import random
import numpy as np
from Graphs import *




N = 9
d = 2
R = regularGraph(N,d)
connectogram(R,title="Degree {} Regular Graph".format(d),curve=1)

N = 7
d = 4
R = regularGraph(N,d)
connectogram(R,title="Degree {} Regular Graph".format(d),curve=0,lineCol='green')