import numpy as np
from scipy.linalg import solve_triangular, qr

def GN(A, r, ell = lambda r : int(np.floor(0.5*r)), seed=42):
    '''
    Generalized Nyström method for sketching a matrix A given r and ell.
    
    Input:
        A    : 2D numpy array, matrix of size (n x m)
        r    : integer, rank of approximation
        ell  : integer or function, oversampling parameter. 
            If ell is given as a function, then it should return an integer.
        seed : integer, seed for reproducability of the algorithm
            
    Output:
        At  : 2D numpy array, approximation to matrix A of rank r using the GN/SKETCHING method.
    '''

    #Set the seed for reproducability.
    np.random.seed(seed)

    #If ell is a function, evaluate it.
    if callable(ell):
        ell = ell(r)
        assert isinstance(ell, int), "Value returned from ell() must be an integer."
    else:
        ell = int(ell)
    
    #Take the size of the matrix A.
    n, m = A.shape

    #Draw random Gaussian matrix X in R^(m x r). Time complexity: O(m*r).
    X = np.random.normal(0, 1, (m, r))

    #Draw random Gaussian matrix Y in R^(n x (r+ell)). Time complexity: O(n*(r+ell)).
    Y = np.random.normal(0, 1, (n, (r+ell)))
    
    #Calculate matrix-matrix products.
    AX = A@X    #size (n x r), time complexity: O(n*m*r)
    YTA = Y.T@A #size ((r+ell) x m), time complexity: O((r+ell)*n*m
    YTAX = Y.T@AX #size ((r+ell) x r), time complexity: O((r+ell)*n*r)
    
    [Q,R] = qr(YTAX, mode='economic') #time complexity: O(2*(r+ell)*r^2)
    
    #Code translated from the MATLAB code in Y. Nakatsukasa's paper, https://arxiv.org/pdf/2009.11392.pdf.
    #https://ch.mathworks.com/matlabcentral/answers/1743765-what-is-the-difference-between-backward-slash-vs-forward-slash-in-matlab?s_tid=srchtitle
    #In MATLAB, the author writes (AX/R), which is equivalent with a triangle solve, which we use from scipy.linalg. 
    B = solve_triangular(R.T, (AX).T, lower=True).T #time complexity: O(r^2)
    C = (Q.T @ YTA).T #time complexity: O((r+ell)*r*m)

    At = B @ C.T

    return At
