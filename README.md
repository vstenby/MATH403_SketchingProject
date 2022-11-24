# Sketching for low-rank matrix approximation

This code was written for our miniproject in the course MATH-403: Low-rank approximation techniques, offered in the fall of 2022 at EPFL. 

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
