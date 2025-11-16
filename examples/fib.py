import math
import time
from omputils import njit, pyomp, omp, omp_pure, use_pyomp, use_pure, use_compiled, use_compiled_types, get_omp_threads

if use_pure():
    omp = omp_pure


@njit
def _pyomp_rfib(n, limit, _deep=0):
    if n < 2:
        return 1
    a = 0
    b = 0
    with pyomp("task if(_deep < limit)"):
        a = _pyomp_rfib(n - 1, limit, _deep + 1)
    with pyomp("task if(_deep < limit)"):
        b = _pyomp_rfib(n - 2, limit, _deep + 1)

    omp("taskwait")
    return a + b


@njit
def _pyomp_fib(n, limit):
    result = 0
    with pyomp("parallel"):
        with pyomp("single"):
            result = _pyomp_rfib(n, limit)
    return result


@omp(compile=use_compiled())
def _omp4py_rfib(n, limit, _deep=0):
    if n < 2:
        return 1
    a = 0
    b = 0
    with omp("task if(_deep < limit)"):
        a = _omp4py_rfib(n - 1, limit, _deep + 1)
    with omp("task if(_deep < limit)"):
        b = _omp4py_rfib(n - 2, limit, _deep + 1)

    omp("taskwait")
    return a + b


@omp(compile=use_compiled())
def _omp4py_fib(n, limit):
    result = 0
    with omp("parallel"):
        with omp("single"):
            result = _omp4py_rfib(n, limit)
    return result


@omp(compile=use_compiled())
def _omp4py_rfib_types(n: int, limit: int, _deep: int = 0) -> int:
    if n < 2:
        return 1
    a: int = 0
    b: int = 0
    with omp("task if(_deep < limit)"):
        a = _omp4py_rfib_types(n - 1, limit, _deep + 1)
    with omp("task if(_deep < limit)"):
        b = _omp4py_rfib_types(n - 2, limit, _deep + 1)

    omp("taskwait")
    return a + b


@omp(compile=use_compiled())
def _omp4py_fib_types(n, limit):
    result = 0
    with omp("parallel"):
        with omp("single"):
            result = _omp4py_rfib_types(n, limit)
    return result


def fib(n=30):
    print(f"fibonacci : n={n}")
    limit = int(math.ceil(math.log2(get_omp_threads())))

    wtime = time.perf_counter()
    if use_pyomp():
        result = _pyomp_fib(n, limit)
    elif use_compiled_types():
        result = _omp4py_fib_types(n, limit)
    else:
        result = _omp4py_fib(n, limit)
    wtime = time.perf_counter() - wtime

    print(f"  Result    : {result}")
    print("Elapsed time : %.6f" % wtime)
