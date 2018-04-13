from Graphs import *

def randAjdMat(N=5):
    R = np.zeros([N,N],dtype="int")
    for x in range(N):
        for y in range(N):
            if x == y:
                continue
            R[x,y] = random.choice([1,1,0,0,0,0,0,0])
    return R


R = randAjdMat(10)
G = connectogram(R)
print(R)

print("\n\nDict created from Graph object")
E = edgeDict(G)
for k,v in E.items():
    print(k,v)

print("\n\nDict created from raw adjacency matrix")
E = edgeDict(R)
for k,v in E.items():
    print(k,v)