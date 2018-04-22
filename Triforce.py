from Graphs import *

D = {'0':[3,5],'1':[3,4],'2':[4,5],'3':[0,1,4,5],'4':[1,2,3,5],'5':[0,2,3,4]}
R = MatFromDict(D)

connectogramUndir(R,['']*6,nodeSize=.1,nodeCol='black')