from scipy.linalg import svd,schur
from numpy.matlib import repmat
import numpy as np

def ave_control(A):
    u, s, vt = svd(A)      # singluar value decomposition
    A = A/(1+s[0])         # s is a eigen-value
    T, U = schur(A,'real') # Schur stability
    midMat = np.power(U,2);
    v = np.matrix(np.diag(T)).transpose()
    P = np.diag(1-np.matmul(v,v.transpose()))
    values = sum(np.divide(midMat,P))
    
    return values

if __name__ == "__main__":
    A = np.array([[0, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0]])
    aa = ave_control(A)
    
    print(aa)