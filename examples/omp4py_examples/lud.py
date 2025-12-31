import time
import numpy as np
from omputils import njit, pyomp, omp, use_pyomp, use_compiled, use_compiled_types

try:
    import cython
except ImportError:
    pass

@njit
def _pyomp_lud(a, l, u):
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


@omp(compile=use_compiled())
def _omp4py_lud(a, l, u):
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

@omp(compile=use_compiled())
def _omp4py_lud_types(a2, l2, u2):
    n:int = a2.shape[0]

    a: cython.double[:, :] = a2
    l: cython.double[:, :] = l2
    u: cython.double[:, :] = u2

    with omp("parallel"):
        k: int
        for k in range(n):
            # U
            with omp("for"):
                for j in range(k, n):
                    u[k][j] = a[k][j]
                    s: int
                    for s in range(k):
                        u[k][j] -= l[k][s] * u[s][j]

            # L
            with omp("for"):
                for i in range(k + 1, n):
                    l[i][k] = a[i][k]
                    s:int
                    for s in range(k):
                        l[i][k] -= l[i][s] * u[s][k]
                    l[i][k] /= u[k][k]

            # diagonal
            with omp("single"):
                l[k][k] = 1.0


def lud(n=1000, seed=0):
    print(f"lud: n={n}, seed={seed}")
    gen = np.random.default_rng(seed)

    l0 = np.tril(gen.random((n, n)))
    np.fill_diagonal(l0, 1)
    u0 = np.triu(np.random.rand(n, n))

    a = np.dot(l0, u0)
    l = np.zeros((n, n))
    u = np.zeros((n, n))

    wtime = time.perf_counter()
    if use_pyomp():
        _pyomp_lud(a, l, u)
    elif use_compiled_types():
        _omp4py_lud_types(a, l, u)
    else:
        _omp4py_lud(a, l, u)
    wtime = time.perf_counter() - wtime
    print("Elapsed time : %.6f" % wtime)
