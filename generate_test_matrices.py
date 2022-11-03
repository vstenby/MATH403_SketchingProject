import numpy as np

def generate_hilbert_matrix(n):
    '''
    Generate the Hilbert matrix.
    '''
    return np.array([[1/(i+j-1) for i in range(1, n+1)] for j in range(1, n+1)])

def generate_A2(n, seed=0):
    '''
    Generate the matrix A2 as described in the problem.
    '''

    #Set the seed.
    np.random.seed(seed)

    d = 0.8 ** np.arange(1, n+1)

    [Q, _] = np.linalg.qr(np.random.normal(0, 1, (n, n)))
    [V, _] = np.linalg.qr(np.random.normal(0, 1, (n, n)))

    A = Q @ np.diag(d) @ V

    return A

def generate_A3(n, seed=0):
    '''
    Generate the matrix A3 as described in the problem.
    '''

    #Set the seed.
    np.random.seed(seed)


    d = np.arange(1, n+1) ** (-1.5)

    [Q, _] = np.linalg.qr(np.random.normal(0, 1, (n, n)))
    [V, _] = np.linalg.qr(np.random.normal(0, 1, (n, n)))

    A = Q @ np.diag(d) @ V

    return A