import numpy as np
import time
from omputils import njit, pyomp, omp


def _partition(array, low, high):
    pivot = array[high]
    i = (low - 1)

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


@omp
def _omp4py_quicksort(array, low, high, limit):
    if high > low:
        pi = _partition(array, low, high)

        with omp("task if(high - low > limit)"):
            _omp4py_quicksort(array, low, pi - 1, limit)

        with omp("task if(high - low > limit)"):
            _omp4py_quicksort(array, pi + 1, high, limit)


@omp
def _omp4py_qsort(array, limit):
    with omp("parallel"):
        with omp("single"):
            _omp4py_quicksort(array, 0, len(array) - 1, limit)


def qsort(n=40000000, limit=1, numba=False):
    print(f"qsort: n={n}, limit={limit}, numba={numba}")
    array = np.random.rand(n)

    wtime = time.perf_counter()
    _omp4py_qsort(array, limit)
    wtime = time.perf_counter() - wtime

    for i in range(1, n):
        assert array[i - 1] < array[i]

    print("Elapsed time      : %.6f" % wtime)
