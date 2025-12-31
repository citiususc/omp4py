import sys
import math
import time
import numpy as np
from omp4py_examples.omputils import njit, pyomp, omp, use_pyomp, use_compiled, use_compiled_types

try:
    import cython
except ImportError:
    pass

try:
    import mpi4py
    import mpi4py.MPI

    has_mpi4py = True
    print("mpi4py found", file=sys.stderr)
except:
    has_mpi4py = False


def check_error(A, x, b):
    r = b - np.dot(A, x)

    return np.linalg.norm(r)


@njit
def _pyomp_jacobi(a, x, b, max_iter, tol):
    n = x.shape[0]

    x_new = np.empty(n)
    norm = 0.0
    iter = 0

    with pyomp("parallel"):
        while iter < max_iter:
            with pyomp("for reduction(+:norm)"):
                for i in range(n):
                    sigma = 0.0
                    for j in range(i):
                        sigma += a[i][j] * x[j]
                    for j in range(i + 1, n):
                        sigma += a[i][j] * x[j]
                    x_new[i] = (b[i] - sigma) / a[i][i]
                    norm += (x_new[i] - x[i]) * (x_new[i] - x[i])

            if math.sqrt(norm) < tol:
                break
            with pyomp("barrier"):
                pass
            with pyomp("single"):
                x, x_new = x_new, x
                norm = 0.0
                iter += 1

    return iter, norm


@omp(compile=use_compiled())
def _omp4py_jacobi(a, x, b, max_iter, tol):
    n = x.shape[0]

    x_new = np.empty(n)
    norm = 0.0
    iter = 0

    with omp("parallel"):
        while iter < max_iter:
            with omp("for reduction(+:norm)"):
                for i in range(n):
                    sigma = 0.0
                    for j in range(i):
                        sigma += a[i][j] * x[j]
                    for j in range(i + 1, n):
                        sigma += a[i][j] * x[j]
                    x_new[i] = (b[i] - sigma) / a[i][i]
                    norm += (x_new[i] - x[i]) * (x_new[i] - x[i])

            if math.sqrt(norm) < tol:
                break
            omp("barrier")
            with omp("single"):
                x, x_new = x_new, x
                norm = 0.0
                iter += 1

    return iter, norm


@omp(compile=use_compiled())
def _omp4py_jacobi_types(a2, x2, b2, max_iter: int, tol: float):
    n: int = x2.shape[0]

    x_new2 = np.empty(n)
    norm: float = 0.0
    iter: int = 0

    a: cython.double[:, :] = a2
    x: cython.double[:] = x2
    b: cython.double[:] = b2
    x_new: cython.double[:] = x_new2

    with omp("parallel"):
        while iter < max_iter:
            with omp("for reduction(+:norm)"):
                for i in range(n):
                    sigma: float = 0.0
                    j: int
                    for j in range(i):
                        sigma += a[i][j] * x[j]
                    for j in range(i + 1, n):
                        sigma += a[i][j] * x[j]
                    x_new[i] = (b[i] - sigma) / a[i][i]
                    norm += (x_new[i] - x[i]) * (x_new[i] - x[i])

            if math.sqrt(norm) < tol:
                break
            omp("barrier")
            with omp("single"):
                x, x_new = x_new, x
                norm = 0.0
                iter += 1

    return iter, norm


@omp(compile=use_compiled() and has_mpi4py)
def _omp4py_mpi_jacobi(a, x, b, max_iter, tol):
    n = x.shape[0]

    x_new = np.empty(n)
    norm = 0.0
    iter = 0

    procs = mpi4py.MPI.COMM_WORLD.size
    rank = mpi4py.MPI.COMM_WORLD.rank
    chunk_size = math.ceil(n / procs)

    start = rank * chunk_size
    end = min(start + chunk_size, n)

    with omp("parallel"):
        while iter < max_iter:
            with omp("for reduction(+:norm)"):
                for i in range(start, end):
                    sigma = 0.0
                    for j in range(i):
                        sigma += a[i][j] * x[j]
                    for j in range(i + 1, n):
                        sigma += a[i][j] * x[j]
                    x_new[i] = (b[i] - sigma) / a[i][i]
                    norm += (x_new[i] - x[i]) * (x_new[i] - x[i])

            with omp("single"):
                norm = mpi4py.MPI.COMM_WORLD.allreduce(norm, op=mpi4py.MPI.SUM)

            if math.sqrt(norm) < tol:
                break
            omp("barrier")
            with omp("single"):
                x_new, x = x, mpi4py.MPI.COMM_WORLD.allgather(x_new[start:end])
                norm = 0.0
                iter += 1

    return iter, norm


@omp(compile=use_compiled() and has_mpi4py)
def _omp4py_mpi_jacobi_types(a2, x2, b2, max_iter: int, tol: float):
    n: int = x2.shape[0]

    x_new2 = np.empty(n)
    norm: float = 0.0
    iter: int = 0

    procs: int = mpi4py.MPI.COMM_WORLD.size
    rank: int = mpi4py.MPI.COMM_WORLD.rank
    chunk_size: int = math.ceil(n / procs)

    start: int = rank * chunk_size
    end: int = min(start + chunk_size, n)

    a: cython.double[:, :] = a2
    x: cython.double[:] = x2
    b: cython.double[:] = b2
    x_new: cython.double[:] = x_new2

    with omp("parallel"):
        while iter < max_iter:
            with omp("for reduction(+:norm)"):
                for i in range(start, end):
                    sigma: float = 0.0
                    j: int
                    for j in range(i):
                        sigma += a[i][j] * x[j]
                    for j in range(i + 1, n):
                        sigma += a[i][j] * x[j]
                    x_new[i] = (b[i] - sigma) / a[i][i]
                    norm += (x_new[i] - x[i]) * (x_new[i] - x[i])

            with omp("single"):
                norm = mpi4py.MPI.COMM_WORLD.allreduce(norm, op=mpi4py.MPI.SUM)

            if math.sqrt(norm) < tol:
                break
            omp("barrier")
            with omp("single"):
                x_new, x = x, mpi4py.MPI.COMM_WORLD.allgather(x_new[start:end])
                norm = 0.0
                iter += 1

    return iter, norm


def jacobi(n=1000, max_iter=1000, tol=1e-6, seed=0):
    procs = mpi4py.MPI.COMM_WORLD.size if has_mpi4py else 1
    if procs > 1:
        raise RuntimeError('PyOmp cannot handle more than one process')

    print(f"jacobi: n={n}, max_iter={max_iter}, tol={tol}, seed={seed}, procs={procs}")
    gen = np.random.default_rng(seed)

    a = gen.random((n, n))
    x = np.zeros(n)
    b = gen.random(n)

    for i in range(n):
        a[i][i] += sum(a[i])

    wtime = time.perf_counter()
    if not use_pyomp():
        if procs == 1:
            if use_compiled_types():
                iters, error = _omp4py_jacobi_types(a, x, b, max_iter, tol)
            else:
                iters, error = _omp4py_jacobi(a, x, b, max_iter, tol)
        else:
            if use_compiled_types():
                iters, error = _omp4py_mpi_jacobi_types(a, x, b, max_iter, tol)
            else:
                iters, error = _omp4py_mpi_jacobi(a, x, b, max_iter, tol)
    else:
        iters, error = _pyomp_jacobi(a, x, b, max_iter, tol)
    wtime = time.perf_counter() - wtime

    print("Total Number of Iterations : %d" % iters)
    print("Residual                   : %.15f" % error)
    print("Solution Error             : %g" % check_error(a, x, b))
    print("Elapsed time               : %.6f" % wtime)
