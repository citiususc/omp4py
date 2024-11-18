import time
import numpy as np
from omputils import njit, pyomp, omp


@njit
def pyomp_lud(a, l, u):
    n = a.shape[0]

    with pyomp("parallel"):
        for k in range(n):
            # U
            with pyomp("for"):
                for j in range(k, n):
                    u[k][j] = a[k][j]
                    for s in range(k):
                        u[k][j] -= l[k][s] * u[s][j]

            # L
            with pyomp("for"):
                for i in range(k + 1, n):
                    l[i][k] = a[i][k]
                    for s in range(k):
                        l[i][k] -= l[i][s] * u[s][k]
                    l[i][k] /= u[k][k]

            # diagonal
            with pyomp("single"):
                l[k][k] = 1.0


@omp
def omp4py_lud(a, l, u):
    n = a.shape[0]

    with omp("parallel"):
        for k in range(n):
            # U
            with omp("for"):
                for j in range(k, n):
                    u[k][j] = a[k][j]
                    for s in range(k):
                        u[k][j] -= l[k][s] * u[s][j]

            # L
            with omp("for"):
                for i in range(k + 1, n):
                    l[i][k] = a[i][k]
                    for s in range(k):
                        l[i][k] -= l[i][s] * u[s][k]
                    l[i][k] /= u[k][k]

            # diagonal
            with omp("single"):
                l[k][k] = 1.0


def lud(n=1000, seed=0, numba=False):
    print(f"lud: n={n}, seed={seed}, numba={numba}")
    gen = np.random.default_rng(seed)

    l0 = np.tril(gen.random((n, n)))
    np.fill_diagonal(l0, 1)
    u0 = np.triu(np.random.rand(n, n))

    a = np.dot(l0, u0)
    l = np.zeros((n, n))
    u = np.zeros((n, n))

    wtime = time.perf_counter()
    omp4py_lud(a, l, u)
    wtime = time.perf_counter() - wtime
    print("Elapsed time : %.6f" % wtime)
