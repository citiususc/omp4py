import sys

try:
    from numba import njit
    from numba.openmp import omp_set_num_threads as pyomp_set_num_threads, openmp_context as pyomp

    has_pyomp = True
    print("pyomp found", file=sys.stderr)
except:
    print("pyomp not found", file=sys.stderr)
    has_pyomp = False
    pyomp = None
    njit = lambda x: x

try:
    from omp4py import omp, omp_set_num_threads as omp4py_set_num_threads

    has_omp4py = True
    print("omp4py found", file=sys.stderr)
except:
    print("omp4py not found", file=sys.stderr)
    has_omp4py = False
    omp = lambda *a, **k: None


def set_omp_threads(n):
    if has_pyomp:
        pyomp_set_num_threads(n)

    if has_omp4py:
        omp4py_set_num_threads(n)

    print("threads : " + str(n))