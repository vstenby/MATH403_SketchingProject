import numpy as np
from scipy.linalg import solve_triangular, qr

def SGN(A, r, ell = None, seed=42):
    '''
    Stabilized Generalized Nyström method for sketching a matrix A given r and ell.
    '''

    #Set the seed for reproducability.
    np.random.seed(seed)

    if ell is None:
        ell = np.floor(0.5 * r).astype(int)
    
    m, n = A.shape

    #Draw random Gaussian matrix X in R^(m x r)
    X = np.random.normal(0, 1, (n, r))

    #Draw random Gaussian matrix Y in R^(n x (r+ell))
    Y = np.random.normal(0, 1, (m, (r+ell)))
    
    #Calculate AX. A is m x n, X is n x r.
    #The time complexity of this operation is O(mnr).
    AX = A@X
    YA = Y.T@A

    [Q,R] = qr(Y.T@AX, mode='economic')

    #https://ch.mathworks.com/matlabcentral/answers/1743765-what-is-the-difference-between-backward-slash-vs-forward-slash-in-matlab?s_tid=srchtitle
    #https://arxiv.org/pdf/2009.11392.pdf
    #In MATLAB, the author writes (AX/R), which is equivalent with a triangle solve. Yeah, this is a bit complicated.
    B = solve_triangular(R.T, (AX).T, lower=True).T
    C = (Q.T @ YA).T

    At = B @ C.T

    #Return the reconstruction error.
    return At