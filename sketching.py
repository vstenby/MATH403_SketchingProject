from generate_test_matrices import generate_hilbert_matrix, generate_A2, generate_A3
from GN import GN
import numpy as np
import argparse 
import inspect

def main():

    parser = argparse.ArgumentParser(description='Sketching Project')

    parser.add_argument('--n', type=int, default=200, help='The size of the matrix.')
    
    #If ell is given, then it should be an integer.
    parser.add_argument('--ell', default = 'lambda r : int(np.floor(0.5*r))', help='The oversampling parameter.')

    #Seeds for the random number generators.
    parser.add_argument('--test_matrix_seed', type=int, default=42, help='Seed for the test matrix.')
    parser.add_argument('--sgn_seed', type=int, default=42, help='Seed for the SGN algorithm.')
    parser.add_argument('--r', type=int, default=10, help='The rank of the matrix.')
    
    #Generate the test matrix.
    parser.add_argument('--test_matrix', type=str, default='hilbert', help='The test matrix to use. Can be hilbert, A2, or A3.', choices=['hilbert', 'A2', 'A3'])

    args = parser.parse_args()
    
    #Generate the test matrix.
    if args.test_matrix == 'hilbert':
        A = generate_hilbert_matrix(args.n)
    elif args.test_matrix == 'A2':
        A = generate_A2(args.n, seed=args.test_matrix_seed)
    elif args.test_matrix == 'A3':
        A = generate_A3(args.n, seed=args.test_matrix_seed)
    else:
        raise Exception('Invalid test matrix.')

    #Calculate the error.
    err = np.linalg.norm(A - GN(A, r=args.r, ell=args.ell, seed=args.sgn_seed), ord='fro')

    print(' Sketching Experiment '.center(80, '-'))
    print(f'Test matrix:                      {args.test_matrix.rjust(46)}')
    print(f'Test matrix seed:                 {args.test_matrix_seed:46}')
    print(f'Size of test matrix:              {args.n:46}')
    print(f'Rank of approximation:            {args.r:46}')
    print(f'Oversampling parameter:{(str(args.ell)).rjust(57)}')
    print(f'Algorithm seed:                   {args.sgn_seed:46}')
    print(f'Reconstruction error (Frobenius): {err:46.2e}')
    return

if __name__ == '__main__':
    main()