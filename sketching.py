from generate_test_matrices import generate_hilbert_matrix, generate_A2, generate_A3
from SGN import SGN
import numpy as np
import argparse 
#import wandb

def main():

    parser = argparse.ArgumentParser(description='Sketching Project')

    parser.add_argument('--n', type=int, default=200, help='The size of the matrix.')
    
    #If ell = None, then ell = floor(0.5 * r).
    parser.add_argument('--ell', type=int, default=None)

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
    err = np.linalg.norm(A - SGN(A, r=args.r, ell=args.ell, seed=args.sgn_seed), ord='fro')
    return

if __name__ == '__main__':
    main()