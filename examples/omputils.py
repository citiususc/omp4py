import sys

_mode = 1
_threads = 1
use_pure = lambda: _mode == 0
use_compiled = lambda: _mode == 2 or _mode == 3
use_compiled_types = lambda: _mode == 3
use_pyomp = lambda: _mode == -1

try:
    from numba.openmp import njit
    from numba.openmp import omp_set_num_threads as pyomp_set_num_threads, openmp_context as pyomp

    has_pyomp = True
    print("pyomp found", file=sys.stderr)
except:
    has_pyomp = False
    pyomp = lambda *a, **k: None
    njit = lambda x: x

try:
    from omp4py import omp, omp_set_num_threads as omp4py_set_num_threads
    from omp4py.pure import omp as omp_pure, omp_set_num_threads as omp4py_pure_set_num_threads

    has_omp4py = True
    print("omp4py found", file=sys.stderr)
    try:
        import pythran

        print("pythran found", file=sys.stderr)
    except ImportError:
        pass
except:
    has_omp4py = False
    omp = lambda *a, **k: (lambda *a, **k: None)
    omp_pure = lambda *a, **k: (lambda *a, **k: None)


def set_omp_threads(n):
    global _threads
    _threads = n
    if has_pyomp:
        pyomp_set_num_threads(n)

    if has_omp4py:
        omp4py_set_num_threads(n)
        omp4py_pure_set_num_threads(n)

    print("threads : " + str(n))


def get_omp_threads():
    return _threads

def set_omp_mode(mode):
    global _mode
    _mode = mode
    print("mode : " + ['pure', 'hybrid', 'compiled', 'compiled with types', 'pyomp'][mode])
