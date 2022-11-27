import numpy as np
from scipy.linalg import solve_triangular, qr

def GN(A, r, ell = lambda r : int(np.floor(0.5*r)), seed=42):
    '''
    Generalized Nyström method for sketching a matrix A given r and ell.
    
    Input:
        A   : 2D numpy array, matrix of size (m x n)
        r   : integer, rank of approximation
        ell : integer or function, oversampling parameter. 
            If ell is given as a function, then it should return an integer.
            
    Output:
        At  : 2D numpy array, approximation to matrix A of rank r using the GN/SKETCHING method.
    '''

    #Set the seed for reproducability.
    np.random.seed(seed)

    #If ell is a function, evaluate it.
    if callable(ell):
        ell = ell(r)
        assert isinstance(ell, int), "Value returned from ell() must be an integer."
    
    #Take the size of the matrix A.
    m, n = A.shape

    #Draw random Gaussian matrix X in R^(m x r)
    X = np.random.normal(0, 1, (n, r))

    #Draw random Gaussian matrix Y in R^(n x (r+ell))
    Y = np.random.normal(0, 1, (m, (r+ell)))
    
    #Calculate AX. A is m x n, X is n x r.
    AX = A@X
    YA = Y.T@A

    [Q,R] = qr(Y.T@AX, mode='economic')
    
    #Code translated from the MATLAB code in Y. Nakatsukasa's paper, https://arxiv.org/pdf/2009.11392.pdf.
    #https://ch.mathworks.com/matlabcentral/answers/1743765-what-is-the-difference-between-backward-slash-vs-forward-slash-in-matlab?s_tid=srchtitle
    #In MATLAB, the author writes (AX/R), which is equivalent with a triangle solve, which we use from scipy.linalg. 
    B = solve_triangular(R.T, (AX).T, lower=True).T
    C = (Q.T @ YA).T

    At = B @ C.T

    return At
