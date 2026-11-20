import numpy as np

from omp4py import *


@omp
def row_norms(matrix, out):
    rows = matrix.shape[0]
    with omp("parallel for num_threads(4)"):
        for i in range(rows):
            out[i] = np.linalg.norm(matrix[i])


rng = np.random.default_rng(0)
m = rng.random((1000, 256))
out = np.empty(1000)
row_norms(m, out)
print(np.allclose(out, np.linalg.norm(m, axis=1)))
