import math
import time
from omputils import njit, pyomp, omp


@njit
def _pyomp_pi(n):
    w = 1.0 / n
    PI = 0.0
    with pyomp("parallel for reduction(+:PI)"):
        for i in range(n):
            local = (i + 0.5) * w
            PI += 4.0 / (1.0 + local * local)
    return PI * w


@omp
def _omp4py_pi(n):
    w = 1.0 / n
    PI = 0.0
    with omp("parallel for reduction(+:PI)"):
        for i in range(n):
            local = (i + 0.5) * w
            PI += 4.0 / (1.0 + local * local)
    return PI * w

@omp
def _omp4py_pi2(n):
    w = 1.0 / n
    PI = 0.0
    with omp("parallel"):
        pi_local = 0.0
        with omp("for"):
            for i in range(n):
                local = (i + 0.5) * w
                pi_local += 4.0 / (1.0 + local * local)
        with omp("critical"):
            PI += pi_local

    return PI * w


def pi(n=2000000000, numba=False):
    print(f"pi: n={n}, numba={numba}")

    wtime = time.perf_counter()
    PI = _pyomp_pi(n) if numba else _omp4py_pi2(n)
    wtime = time.perf_counter() - wtime

    error = math.fabs(PI - math.pi)
    print(f"  Estimate       : {round(PI, 16)}")
    print(f"  Error          : {round(error, 16)}")
    print("Elapsed time      : %.6f" % wtime)
