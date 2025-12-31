import os
import sys

__all__ = ["use_compiled", "use_compiled_types", "use_pyomp", "pyomp", "njit", "omp",
           "set_omp_threads", "get_omp_threads", "set_omp_mode"]

_mode = 1
_threads = 1
use_compiled = lambda: _mode == 2 or _mode == 3
use_compiled_types = lambda: _mode == 3
use_pyomp = lambda: _mode == -1

pyomp = lambda *a, **k: None
njit = lambda x: x
pyomp_set_num_threads = lambda n: None

omp = lambda *a, **k: (lambda *a, **k: None)
omp4py_set_num_threads = lambda n: None


def set_omp_threads(n):
    global _threads
    _threads = n
    pyomp_set_num_threads(n)
    omp4py_set_num_threads(n)

    print("threads : " + str(n))


def get_omp_threads():
    return _threads


def set_omp_mode(mode):
    global _mode, pyomp, njit, omp, pyomp_set_num_threads, omp4py_set_num_threads
    _mode = mode

    if mode == 0:
        os.environ["OMP4PY_PURE"] = "1"

    if mode == -1:
        from numba.openmp import njit
        from numba.openmp import omp_set_num_threads as pyomp_set_num_threads, openmp_context as pyomp
    else:
        from omp4py import omp, omp_set_num_threads as omp4py_set_num_threads
        try:
            import pythran
            print("pythran found", file=sys.stderr)
        except ImportError:
            pass

    print("mode : " + ['pure', 'hybrid', 'compiled', 'compiled with types', 'pyomp'][mode])
