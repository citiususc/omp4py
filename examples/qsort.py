import numpy as np
import time
from omputils import njit, pyomp, omp, omp_pure, use_pyomp, use_pure, use_compiled, use_compiled_types

if use_pure():
    omp = omp_pure

try:
    import cython
except ImportError:
    pass


def _partition(array, low, high):
    pivot = array[high]
    i = low - 1

    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1


@njit
def _pyomp_quicksort(array, low, high, limit):
    if high > low:
        pi = _partition(array, low, high)

        with pyomp("task if(high - low > limit)"):
            _pyomp_quicksort(array, low, pi - 1, limit)

        with pyomp("task if(high - low > limit)"):
            _pyomp_quicksort(array, pi + 1, high, limit)


@njit
def _pyomp_qsort(array, limit):
    with pyomp("parallel"):
        with pyomp("single"):
            _pyomp_quicksort(array, 0, len(array) - 1, limit)


@omp(compile=use_compiled())
def _omp4py_quicksort(array, low, high, limit):
    if high > low:
        pi = _partition(array, low, high)

        with omp("task if(high - low > limit)"):
            _omp4py_quicksort(array, low, pi - 1, limit)

        with omp("task if(high - low > limit)"):
            _omp4py_quicksort(array, pi + 1, high, limit)


@omp(compile=use_compiled())
def _omp4py_qsort(array, limit):
    with omp("parallel"):
        with omp("single"):
            _omp4py_quicksort(array, 0, len(array) - 1, limit)


@omp(compile=use_compiled())
def _omp4py_quicksort_types(array: cython.double[:], low: int, high: int, limit: int):
    if high > low:

        pivot: float = array[high]
        i: int = low - 1
        j: int
        for j in range(low, high):
            if array[j] <= pivot:
                i += 1
                array[i], array[j] = array[j], array[i]
        array[i + 1], array[high] = array[high], array[i + 1]
        pi: int = i + 1

        with omp("task if(high - low > limit)"):
            _omp4py_quicksort(array, low, pi - 1, limit)

        with omp("task if(high - low > limit)"):
            _omp4py_quicksort(array, pi + 1, high, limit)


@omp(compile=use_compiled())
def _omp4py_qsort_types(array2, limit: int):
    with omp("parallel"):
        with omp("single"):
            array: cython.double[:] = array2
            _omp4py_quicksort(array, 0, len(array) - 1, limit)


def qsort(n=40000000, limit=100000):
    print(f"qsort: n={n}, limit={limit}")
    array = np.random.rand(n)

    wtime = time.perf_counter()
    if use_pyomp():
        _pyomp_qsort(array, limit)
    elif use_compiled_types():
        _omp4py_qsort(array, limit)
    else:
        _omp4py_qsort_types(array, limit)
    wtime = time.perf_counter() - wtime

    for i in range(1, n):
        assert array[i - 1] < array[i]

    print("Elapsed time      : %.6f" % wtime)
