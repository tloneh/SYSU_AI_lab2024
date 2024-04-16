import copy

def MatrixAdd(A, B):
    n = len(A)
    if n == 0:
        return -1
    C = copy.deepcopy(A) #deepcopy
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] += B[i][j] # add
    return C
    
def MatrixMul(A, B):    
    n = len(A)
    if n == 0:
        return -1
    C = copy.deepcopy(A) #deepcopy
    for i in range (0, n):
        for j in range(0, n):
            for k in range(0, n):
                C[i][j] = A[i][k] * B[k][j] # mul
    return C

#test
A = [[1, 2], [3, 4]]
B = [[3, 5], [6, 8]]
print(MatrixAdd(A, B))
print(MatrixMul(A, B))
print(A) # chack A is invariable
print(B)
