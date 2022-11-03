import numpy as np
from scipy.linalg import solve_triangular, qr

def hilbert_matrix(n):
    if n == 1:
        return np.array([[1]])
    else:
        H = hilbert_matrix(n-1)
        return np.block([[H, H+1], [H+1, H]])

def sketching(A, r, ell = None):
    '''
    (Stabilized) Generalized Nyström.
    '''

    if ell is None:
        ell = np.ceil(0.5 * r).astype(int)
    
    m, n = A.shape

    #Draw random Gaussian matrix X in R^(m x r)
    X = np.random.normal(0, 1, (n, r))

    #Draw random Gaussian matrix Y in R^(n x (r+ell))
    Y = np.random.normal(0, 1, (m, (r+ell)))
    
    AX = A@X
    YA = Y.T@A

    [Q,R] = qr(Y.T@AX, mode='economic')

    #https://ch.mathworks.com/matlabcentral/answers/1743765-what-is-the-difference-between-backward-slash-vs-forward-slash-in-matlab?s_tid=srchtitle
    #https://arxiv.org/pdf/2009.11392.pdf
    #In MATLAB, the author writes (AX/R), which is equivalent with a triangle solve. Yeah, this is a bit complicated.
    B = solve_triangular(R.T, (AX).T, lower=True).T
    C = (Q.T @ YA).T

    #At = B @ C.T

    return 

def main():
    A = hilbert_matrix(100)
    
    sketching(A, 9)

    #Print the Frobenius norm of the difference.
    #print(np.linalg.norm(A - At, ord='fro'))

    return


if __name__ == '__main__':
    main()