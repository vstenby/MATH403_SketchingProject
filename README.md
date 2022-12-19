# Sketching for low-rank matrix approximation

This code was written for our miniproject in the course MATH-403: Low-rank approximation techniques, offered in the fall of 2022 at EPFL. 

To reproduce the results in the reports, check out the `Experiments.ipynb` notebook, which we ran with the following specs:

```
Python implementation: CPython
Python version       : 3.9.13
IPython version      : 8.4.0

numpy     : 1.22.3
scipy     : 1.8.0
matplotlib: 3.6.2
tqdm      : 4.63.0

Compiler    : GCC 11.2.0
OS          : Linux
Release     : 3.10.0-1160.80.1.el7.x86_64
Machine     : x86_64
Processor   : x86_64
CPU cores   : 20
Architecture: 64bit
```

## Files

#### `generate_test_matrices.py`

`generate_test_matrices.py` contains three functions, which were used to construct the test matrices for the project. 

`generate_hilbert_matrix(n)` generates a $n \times n$ Hilbert matrix $\mathbf{A}_1$, where:

$$
\mathbf{A}_1(i,j) = \frac{1}{i+j-1}
$$

`generate_A2(n)` generates a $n \times n$ test matrix $\mathbf{A}_2$ as follows:

```python
d = 0.8 ** np.arange(1, n+1)
[Q, _] = np.linalg.qr(np.random.normal(0, 1, (n,n)))
[V, _] = np.linalg.qr(np.random.normal(0, 1, (n,n)))
A = Q @ np.diag(d) @ V
```

`generate_A3(n)` generates a $n \times n$ test matrix $\mathbf{A}_3$ as follows:

```python
d = np.arange(1, n+1) ** (-1.5)
[Q, _] = np.linalg.qr(np.random.normal(0, 1, (n,n)))
[V, _] = np.linalg.qr(np.random.normal(0, 1, (n,n)))
A = Q @ np.diag(d) @ V
```

### `GN.py`

Implements the Stabilized Generalized Nyström algorithm, `SKETCHING`. 

### `sketching.py`

Implementation of a run for a given set of parameters, callable from a terminal. 

For example, to do $\texttt{SKETCHING}(\mathbf{A}_2 \in \mathbb{R}^{100 \times 100}, r = 25, \ell = 5)$ with default GN and $\mathbf{A}_2$ seed, write the following in your terminal:

```
python sketching.py --test_matrix A2 --n 100 --r 25 --ell 5
```

it gives the following output:

```
----------------------------- Sketching Experiment -----------------------------
Test matrix:                                                                  A2
Test matrix seed:                                                             42
Size of test matrix:                                                         100
Rank of approximation:                                                        25
Oversampling parameter:                                                        5
Algorithm seed:                                                               42
Reconstruction error (Frobenius):                                       2.62e-02
```

### `Experiments.ipynb`

Notebook containing experiments done for the report. 
